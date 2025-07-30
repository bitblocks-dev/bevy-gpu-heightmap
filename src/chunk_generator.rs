use bevy::asset::{RenderAssetUsages, load_internal_asset, weak_handle};
use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{self, RenderGraph, RenderLabel};
use bevy::render::render_resource::binding_types::{storage_buffer, uniform_buffer};
use bevy::render::render_resource::encase::private::{ReadFrom as _, Reader};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BufferUsages,
    CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
    ShaderStages, ShaderType, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::storage::{GpuShaderStorageBuffer, ShaderStorageBuffer};
use bevy::render::{Render, RenderApp, RenderSet};

use crate::terrain_sampler::DensitySampler;

#[derive(Debug, Clone, ShaderType)]
#[repr(C)]
struct Vertex {
    position: Vec3,
    _padding1: f32,
    normal: Vec3,
    _padding2: f32,
}

#[derive(Debug, Clone, ShaderType)]
#[repr(C)]
struct Triangle {
    vertex_a: u32,
    vertex_b: u32,
    vertex_c: u32,
}

#[derive(Resource, Debug, Clone)]
pub struct ChunkGenerator<T> {
    pub surface_threshold: f32,
    pub num_voxels_per_axis: u32,
    pub chunk_size: f32,
    pub terrain_sampler: T,
}

impl<T> ChunkGenerator<T> {
    pub fn num_samples_per_axis(&self) -> u32 {
        self.num_voxels_per_axis + 3 // We sample the next chunk over too for normals
    }

    pub fn max_num_vertices(&self) -> u64 {
        self.max_num_triangles() * 3
    }

    pub fn max_num_triangles(&self) -> u64 {
        (self.num_voxels_per_axis as u64).pow(3) * 5
    }
}

pub struct SampleContext<'a, T> {
    pub world_position: Vec3,
    pub local_position: Vec3,
    pub generator: &'a ChunkGenerator<T>,
}

impl<T: DensitySampler> ChunkGenerator<T> {
    pub fn voxel_size(&self) -> f32 {
        self.chunk_size / self.num_voxels_per_axis as f32
    }

    pub fn sample_density(&self, chunk_id: IVec3, sample_id: IVec3) -> f32 {
        self.terrain_sampler.sample_density(SampleContext {
            world_position: self.coord_to_world(chunk_id, sample_id),
            local_position: self.coord_to_local(sample_id),
            generator: self,
        })
    }

    fn coord_to_local(&self, voxel_id: IVec3) -> Vec3 {
        voxel_id.as_vec3() * self.voxel_size()
    }

    fn coord_to_world(&self, chunk_id: IVec3, voxel_id: IVec3) -> Vec3 {
        (chunk_id * self.num_voxels_per_axis as i32 + voxel_id).as_vec3() * self.voxel_size()
    }
}

pub struct MarchingCubesPlugin<Sampler, Material> {
    _marker: std::marker::PhantomData<(Sampler, Material)>,
}

impl<Sampler, Material> Default for MarchingCubesPlugin<Sampler, Material> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Sampler: DensitySampler + Send + Sync + 'static, Material: Asset + bevy::prelude::Material>
    MarchingCubesPlugin<Sampler, Material>
{
    fn update_chunk_loaders(
        generator: Res<ChunkGenerator<Sampler>>,
        mut chunk_loaders: Query<(&mut ChunkLoader, &GlobalTransform), Changed<GlobalTransform>>,
    ) {
        for (mut chunk_loader, transform) in chunk_loaders.iter_mut() {
            let chunk_position = (transform.translation() / generator.chunk_size)
                .floor()
                .as_ivec3();

            // Properly update change detection
            if chunk_loader.position != chunk_position {
                chunk_loader.position = chunk_position;
            }
        }
    }

    fn queue_chunks(
        mut chunk_loading: ResMut<ChunkLoading<Sampler>>,
        chunk_loaders: Query<&ChunkLoader, Changed<ChunkLoader>>,
    ) {
        for chunk_loader in chunk_loaders.iter() {
            for x in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
                for y in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
                    for z in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
                        let chunk_position = chunk_loader.position + IVec3::new(x, y, z);

                        if !chunk_loading.loaded_chunks.contains(&chunk_position) {
                            chunk_loading.loaded_chunks.insert(chunk_position);
                            chunk_loading.chunks_to_load.push(chunk_position);
                            info!("Queued chunk for loading: {chunk_position:?}");
                        }
                    }
                }
            }
        }
    }

    fn finish_chunks(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut current_chunk_data: ResMut<CurrentChunkData>,
        mut chunk_loading: ResMut<ChunkLoading<Sampler>>,
        mut next_chunk_data: ResMut<NextChunkData>,
        generator: Res<ChunkGenerator<Sampler>>,
        material: Res<ChunkMaterial<Sampler, Material>>,
    ) {
        let Some(chunk_position) = chunk_loading.current_chunk else {
            return;
        };
        let Some(vertices) = &current_chunk_data.vertices else {
            return;
        };
        let Some(vertices_len) = current_chunk_data.vertices_len else {
            return;
        };
        let Some(triangles) = &current_chunk_data.triangles else {
            return;
        };
        let Some(triangles_len) = current_chunk_data.triangles_len else {
            return;
        };

        info!(
            "Finished chunk {chunk_position:?} with {vertices_len} vertices and {triangles_len} triangles"
        );

        let vertices: Vec<u8> = vertices
            .iter()
            .take(size_of::<Vertex>() * vertices_len as usize)
            .cloned()
            .collect();
        let mut vertices_reader =
            Reader::new::<Vec<Vertex>>(vertices, 0).expect("Failed to create Reader");
        let mut vertices = Vec::<Vertex>::new();
        vertices.read_from(&mut vertices_reader);

        let triangles: Vec<u8> = triangles
            .iter()
            .take(size_of::<Triangle>() * triangles_len as usize)
            .cloned()
            .collect();
        let mut triangles_reader =
            Reader::new::<Vec<Triangle>>(triangles, 0).expect("Failed to create Reader");
        let mut triangles = Vec::<Triangle>::new();
        triangles.read_from(&mut triangles_reader);

        let mesh = Mesh::new(
            bevy::render::mesh::PrimitiveTopology::TriangleList,
            bevy::render::render_asset::RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_indices(bevy::render::mesh::Indices::U32(
            triangles
                .iter()
                .flat_map(|t| [t.vertex_c, t.vertex_b, t.vertex_a])
                .collect(),
        ))
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vertices.iter().map(|v| v.position).collect::<Vec<_>>(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            vertices.iter().map(|v| v.normal).collect::<Vec<_>>(),
        );

        commands.spawn((
            Name::new(format!("Chunk {chunk_position:?}")),
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(material.material.clone()),
            Transform::from_translation(chunk_position.as_vec3() * generator.chunk_size),
        ));

        chunk_loading.current_chunk = None;
        current_chunk_data.vertices = None;
        current_chunk_data.vertices_len = None;
        current_chunk_data.triangles = None;
        current_chunk_data.triangles_len = None;
        next_chunk_data.should_run = false;
    }

    fn start_chunks(
        mut next_chunk_data: ResMut<NextChunkData>,
        mut chunk_loading: ResMut<ChunkLoading<Sampler>>,
        generator: Res<ChunkGenerator<Sampler>>,
        mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
        storage_buffers: Res<MarchingCubesStorageBuffers>,
        mut commands: Commands,
    ) {
        if chunk_loading.current_chunk.is_some() {
            return;
        }

        let Some(chunk_position) = chunk_loading.chunks_to_load.pop() else {
            return;
        };

        chunk_loading.current_chunk = Some(chunk_position);

        let mut densities = Vec::<f32>::new();
        for x in 0..generator.num_samples_per_axis() {
            for y in 0..generator.num_samples_per_axis() {
                for z in 0..generator.num_samples_per_axis() {
                    densities.push(
                        generator.sample_density(
                            chunk_position,
                            IVec3::new(x as i32, y as i32, z as i32),
                        ),
                    );
                }
            }
        }

        buffers
            .get_mut(&storage_buffers.densities)
            .unwrap()
            .set_data(densities);
        buffers
            .get_mut(&storage_buffers.out_vertices_len)
            .unwrap()
            .set_data(0);
        buffers
            .get_mut(&storage_buffers.out_triangles_len)
            .unwrap()
            .set_data(0);

        next_chunk_data.should_run = true;
        next_chunk_data.num_voxels_per_axis = generator.num_voxels_per_axis;
        next_chunk_data.num_samples_per_axis = generator.num_samples_per_axis();
        next_chunk_data.chunk_size = generator.chunk_size;
        next_chunk_data.surface_threshold = generator.surface_threshold;

        let out_vertices_rb = commands
            .spawn(Readback::buffer(storage_buffers.out_vertices.clone()))
            .id();
        commands.entity(out_vertices_rb).observe(
            move |trigger: Trigger<ReadbackComplete>,
                  mut data: ResMut<CurrentChunkData>,
                  mut commands: Commands| {
                data.vertices = Some(trigger.event().0.clone());
                commands.entity(out_vertices_rb).despawn();
            },
        );
        let out_vertices_len_rb = commands
            .spawn(Readback::buffer(storage_buffers.out_vertices_len.clone()))
            .id();
        commands.entity(out_vertices_len_rb).observe(
            move |trigger: Trigger<ReadbackComplete>,
                  mut data: ResMut<CurrentChunkData>,
                  mut commands: Commands| {
                data.vertices_len = Some(trigger.event().to_shader_type());
                commands.entity(out_vertices_len_rb).despawn();
            },
        );
        let out_triangles_rb = commands
            .spawn(Readback::buffer(storage_buffers.out_triangles.clone()))
            .id();
        commands.entity(out_triangles_rb).observe(
            move |trigger: Trigger<ReadbackComplete>,
                  mut data: ResMut<CurrentChunkData>,
                  mut commands: Commands| {
                data.triangles = Some(trigger.event().0.clone());
                commands.entity(out_triangles_rb).despawn();
            },
        );
        let out_triangles_len_rb = commands
            .spawn(Readback::buffer(storage_buffers.out_triangles_len.clone()))
            .id();
        commands.entity(out_triangles_len_rb).observe(
            move |trigger: Trigger<ReadbackComplete>,
                  mut data: ResMut<CurrentChunkData>,
                  mut commands: Commands| {
                data.triangles_len = Some(trigger.event().to_shader_type());
                commands.entity(out_triangles_len_rb).despawn();
            },
        );
    }

    fn setup(
        mut commands: Commands,
        mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
        chunk_generator: Res<ChunkGenerator<Sampler>>,
    ) {
        let mut densities = ShaderStorageBuffer::with_size(
            size_of::<f32>() * chunk_generator.num_samples_per_axis().pow(3) as usize,
            RenderAssetUsages::default(),
        );
        densities.buffer_description.usage |= BufferUsages::COPY_SRC;
        let densities = buffers.add(densities);

        let mut out_vertices = ShaderStorageBuffer::with_size(
            size_of::<Vertex>() * chunk_generator.max_num_vertices() as usize,
            RenderAssetUsages::default(),
        );
        out_vertices.buffer_description.usage |= BufferUsages::COPY_SRC;
        let out_vertices = buffers.add(out_vertices);

        let mut out_vertices_len =
            ShaderStorageBuffer::with_size(size_of::<u32>(), RenderAssetUsages::default());
        out_vertices_len.buffer_description.usage |= BufferUsages::COPY_SRC;
        let out_vertices_len = buffers.add(out_vertices_len);

        let mut out_triangles = ShaderStorageBuffer::with_size(
            size_of::<Triangle>() * chunk_generator.max_num_triangles() as usize,
            RenderAssetUsages::default(),
        );
        out_triangles.buffer_description.usage |= BufferUsages::COPY_SRC;
        let out_triangles = buffers.add(out_triangles);

        let mut out_triangles_len =
            ShaderStorageBuffer::with_size(size_of::<u32>(), RenderAssetUsages::default());
        out_triangles_len.buffer_description.usage |= BufferUsages::COPY_SRC;
        let out_triangles_len = buffers.add(out_triangles_len);

        // This is just a simple way to pass the buffer handle to the render app for our compute node
        commands.insert_resource(MarchingCubesStorageBuffers {
            densities,
            out_vertices,
            out_vertices_len,
            out_triangles,
            out_triangles_len,
        });
    }

    fn prepare_uniform_buffers(
        render_device: Res<RenderDevice>,
        render_queue: Res<RenderQueue>,
        mut uniform_buffers: ResMut<MarchingCubesUniformBuffers>,
        data: Res<NextChunkData>,
    ) {
        let num_voxels_per_axis = uniform_buffers.num_voxels_per_axis.get_mut();
        *num_voxels_per_axis = data.num_voxels_per_axis;
        uniform_buffers
            .num_voxels_per_axis
            .write_buffer(&render_device, &render_queue);

        let num_samples_per_axis = uniform_buffers.num_samples_per_axis.get_mut();
        *num_samples_per_axis = data.num_samples_per_axis;
        uniform_buffers
            .num_samples_per_axis
            .write_buffer(&render_device, &render_queue);

        let chunk_size = uniform_buffers.chunk_size.get_mut();
        *chunk_size = data.chunk_size;
        uniform_buffers
            .chunk_size
            .write_buffer(&render_device, &render_queue);

        let surface_threshold = uniform_buffers.surface_threshold.get_mut();
        *surface_threshold = data.surface_threshold;
        uniform_buffers
            .surface_threshold
            .write_buffer(&render_device, &render_queue);
    }
}

impl<Sampler: DensitySampler + Send + Sync + 'static, Material: Asset + bevy::prelude::Material>
    Plugin for MarchingCubesPlugin<Sampler, Material>
{
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<MarchingCubesStorageBuffers>::default(),
            ExtractResourcePlugin::<NextChunkData>::default(),
        ))
        .add_systems(Startup, Self::setup)
        .add_systems(
            Update,
            (
                Self::update_chunk_loaders,
                Self::finish_chunks,
                Self::queue_chunks,
                Self::start_chunks,
            )
                .chain(),
        )
        .init_resource::<ChunkLoading<Sampler>>()
        .init_resource::<CurrentChunkData>()
        .init_resource::<NextChunkData>();

        load_internal_asset!(
            app,
            MARCHING_CUBES_SHADER,
            "../assets/marching_cubes.wgsl",
            Shader::from_wgsl
        );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<MarchingCubesComputePipeline>()
            .init_resource::<MarchingCubesUniformBuffers>()
            .add_systems(
                Render,
                (
                    prepare_bind_group
                        .in_set(RenderSet::PrepareBindGroups)
                        // We don't need to recreate the bind group every frame
                        .run_if(not(resource_exists::<MarchingCubesBindGroup>)),
                    Self::prepare_uniform_buffers.in_set(RenderSet::PrepareResources),
                ),
            );

        // Add the compute node as a top level node to the render graph
        // This means it will only execute once per frame
        render_app
            .world_mut()
            .resource_mut::<RenderGraph>()
            .add_node(MarchingCubesComputeNodeLabel, MarchingCubesComputeNode);
    }
}

const WORKGROUP_SIZE: u32 = 8;

#[derive(Resource, Debug)]
struct ChunkLoading<T> {
    loaded_chunks: HashSet<IVec3>,
    chunks_to_load: Vec<IVec3>,
    current_chunk: Option<IVec3>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Default for ChunkLoading<T> {
    fn default() -> Self {
        Self {
            loaded_chunks: HashSet::default(),
            chunks_to_load: Vec::new(),
            current_chunk: None,
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Component, Default, Debug)]
pub struct ChunkLoader {
    pub position: IVec3,
    pub loading_radius: i32,
}

impl ChunkLoader {
    pub fn new(loading_radius: i32) -> Self {
        Self {
            position: IVec3::ZERO,
            loading_radius,
        }
    }
}

#[derive(Resource, Debug)]
pub struct ChunkMaterial<Sampler, Material: Asset> {
    pub material: Handle<Material>,
    _marker: std::marker::PhantomData<Sampler>,
}

impl<Sampler, Material: Asset> ChunkMaterial<Sampler, Material> {
    pub fn new(material: Handle<Material>) -> Self {
        Self {
            material,
            _marker: std::marker::PhantomData,
        }
    }
}

// Resides on the main app so it can be extracted
#[derive(Resource, ExtractResource, Clone)]
struct MarchingCubesStorageBuffers {
    // Include this one, we're gonna stick it in another node later
    densities: Handle<ShaderStorageBuffer>,
    out_vertices: Handle<ShaderStorageBuffer>,
    out_vertices_len: Handle<ShaderStorageBuffer>,
    out_triangles: Handle<ShaderStorageBuffer>,
    out_triangles_len: Handle<ShaderStorageBuffer>,
}

// Resides on the render app
#[derive(Resource, Default)]
struct MarchingCubesUniformBuffers {
    num_voxels_per_axis: UniformBuffer<u32>,
    num_samples_per_axis: UniformBuffer<u32>,
    chunk_size: UniformBuffer<f32>,
    surface_threshold: UniformBuffer<f32>,
}

#[derive(Resource)]
struct MarchingCubesBindGroup(BindGroup);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<MarchingCubesComputePipeline>,
    render_device: Res<RenderDevice>,
    storage_buffers: Res<MarchingCubesStorageBuffers>,
    uniform_buffers: Res<MarchingCubesUniformBuffers>,
    buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
) {
    let densities = buffers.get(&storage_buffers.densities).unwrap();
    let num_voxels_per_axis = &uniform_buffers.num_voxels_per_axis;
    let num_samples_per_axis = &uniform_buffers.num_samples_per_axis;
    let chunk_size = &uniform_buffers.chunk_size;
    let surface_threshold = &uniform_buffers.surface_threshold;
    let out_vertices = buffers.get(&storage_buffers.out_vertices).unwrap();
    let out_vertices_len = buffers.get(&storage_buffers.out_vertices_len).unwrap();
    let out_triangles = buffers.get(&storage_buffers.out_triangles).unwrap();
    let out_triangles_len = buffers.get(&storage_buffers.out_triangles_len).unwrap();

    let bind_group = render_device.create_bind_group(
        None,
        &pipeline.layout,
        &BindGroupEntries::sequential((
            densities.buffer.as_entire_buffer_binding(),
            num_voxels_per_axis,
            num_samples_per_axis,
            chunk_size,
            surface_threshold,
            out_vertices.buffer.as_entire_buffer_binding(),
            out_vertices_len.buffer.as_entire_buffer_binding(),
            out_triangles.buffer.as_entire_buffer_binding(),
            out_triangles_len.buffer.as_entire_buffer_binding(),
        )),
    );
    commands.insert_resource(MarchingCubesBindGroup(bind_group));
}

const MARCHING_CUBES_SHADER: Handle<Shader> = weak_handle!("905ba5ed-f841-4ffe-a169-d51b37b6f4a0");

#[derive(Resource)]
struct MarchingCubesComputePipeline {
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for MarchingCubesComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            None,
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<Vec<u32>>(false),
                    uniform_buffer::<u32>(false),
                    uniform_buffer::<u32>(false),
                    uniform_buffer::<f32>(false),
                    uniform_buffer::<f32>(false),
                    storage_buffer::<Vec<Vertex>>(false),
                    storage_buffer::<u32>(false),
                    storage_buffer::<Vec<Triangle>>(false),
                    storage_buffer::<u32>(false),
                ),
            ),
        );
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Marching cubes compute shader".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: MARCHING_CUBES_SHADER.clone(),
            shader_defs: Vec::new(),
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });
        MarchingCubesComputePipeline { layout, pipeline }
    }
}

/// Label to identify the node in the render graph
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct MarchingCubesComputeNodeLabel;

/// The node that will execute the compute shader
#[derive(Default)]
struct MarchingCubesComputeNode;

impl render_graph::Node for MarchingCubesComputeNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<MarchingCubesComputePipeline>();
        let bind_group = world.resource::<MarchingCubesBindGroup>();
        let next_chunk_data = world.resource::<NextChunkData>();

        if let Some(init_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline)
            && next_chunk_data.should_run
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Marching cubes compute pass"),
                        ..default()
                    });

            pass.set_bind_group(0, &bind_group.0, &[]);
            pass.set_pipeline(init_pipeline);
            let dispatch_size =
                (next_chunk_data.num_voxels_per_axis as f32 / WORKGROUP_SIZE as f32).ceil() as u32;
            pass.dispatch_workgroups(dispatch_size, dispatch_size, dispatch_size);
        }
        Ok(())
    }
}

#[derive(Resource, Default)]
struct CurrentChunkData {
    vertices: Option<Vec<u8>>,
    vertices_len: Option<u32>,
    triangles: Option<Vec<u8>>,
    triangles_len: Option<u32>,
}

#[derive(Resource, ExtractResource, Default, Clone)]
struct NextChunkData {
    should_run: bool,
    num_voxels_per_axis: u32,
    num_samples_per_axis: u32,
    chunk_size: f32,
    surface_threshold: f32,
}
