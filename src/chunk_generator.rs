use std::time::Duration;

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy_app_compute::prelude::*;
use bytemuck::{Pod, Zeroable};

use crate::Chunk;

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Vertex {
    position: Vec3,
    _padding1: f32,
    normal: Vec3,
    _padding2: f32,
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Triangle {
    vertex_a: u32,
    vertex_b: u32,
    vertex_c: u32,
}

#[derive(Resource, Debug, Clone)]
pub struct ChunkGenerator<T> {
    surface_threshold: f32,
    num_voxels_per_axis: u32,
    chunk_size: f32,
    loaded_chunks: HashMap<IVec3, LoadState>,
    chunks_to_load: Vec<IVec3>,
    current_chunk: Option<IVec3>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> ChunkGenerator<T> {
    pub fn new(surface_threshold: f32, num_voxels_per_axis: u32, chunk_size: f32) -> Self {
        Self {
            surface_threshold,
            num_voxels_per_axis,
            chunk_size,
            loaded_chunks: HashMap::default(),
            chunks_to_load: Vec::new(),
            current_chunk: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn num_samples_per_axis(&self) -> u32 {
        self.num_voxels_per_axis + 3 // We sample the next chunk over too for normals
    }

    pub fn max_num_vertices(&self) -> u64 {
        self.max_num_triangles() * 3
    }

    pub fn max_num_triangles(&self) -> u64 {
        (self.num_voxels_per_axis as u64).pow(3) * 5
    }

    pub fn voxel_size(&self) -> f32 {
        self.chunk_size / self.num_voxels_per_axis as f32
    }

    pub fn is_chunk_marked(&self, chunk_position: &IVec3) -> bool {
        self.loaded_chunks.contains_key(chunk_position)
    }

    pub fn is_chunk_generated(&self, chunk_position: &IVec3) -> bool {
        matches!(
            self.loaded_chunks.get(chunk_position),
            Some(LoadState::Finished)
        )
    }

    pub fn is_chunk_with_position_marked(&self, position: Vec3) -> bool {
        self.is_chunk_marked(&self.position_to_chunk(position))
    }

    pub fn is_chunk_with_position_generated(&self, position: Vec3) -> bool {
        self.is_chunk_generated(&self.position_to_chunk(position))
    }

    pub fn position_to_chunk(&self, position: Vec3) -> IVec3 {
        (position / self.chunk_size).floor().as_ivec3()
    }

    pub fn chunk_to_position(&self, chunk: IVec3) -> Vec3 {
        chunk.as_vec3() * self.chunk_size
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LoadState {
    Loading,
    Finished,
}

pub struct SampleContext<'a, T> {
    pub world_position: Vec3,
    pub local_position: Vec3,
    pub generator: &'a ChunkGenerator<T>,
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

impl<Sampler: ComputeShader + Send + Sync + 'static, Material: Asset + bevy::prelude::Material>
    MarchingCubesPlugin<Sampler, Material>
{
    fn update_chunk_loaders(
        generator: Res<ChunkGenerator<Sampler>>,
        mut chunk_loaders: Query<
            (&mut ChunkLoader<Sampler>, &GlobalTransform),
            Changed<GlobalTransform>,
        >,
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
        mut generator: ResMut<ChunkGenerator<Sampler>>,
        chunk_loaders: Query<&ChunkLoader<Sampler>, Changed<ChunkLoader<Sampler>>>,
    ) {
        for chunk_loader in chunk_loaders.iter() {
            let mut load_order = Vec::new();
            for x in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
                for y in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
                    for z in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
                        load_order.push(Vec3::new(x as f32, y as f32, z as f32));
                    }
                }
            }

            load_order.sort_by(|a, b| {
                // Why does copilot insist on using length_squared() here? Square roots aren't expensive
                // Anyway, sort backwards so that the closest chunks are loaded first
                b.length_squared()
                    .partial_cmp(&a.length_squared())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for offset in load_order {
                let chunk_position = chunk_loader.position + offset.as_ivec3();
                if !generator.is_chunk_marked(&chunk_position) {
                    generator
                        .loaded_chunks
                        .insert(chunk_position, LoadState::Loading);
                    generator.chunks_to_load.push(chunk_position);
                    // info!("Queued chunk for loading: {chunk_position:?}");
                }
            }
        }
    }

    fn finish_chunks(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        compute_worker: Res<AppComputeWorker<MarchingCubesComputeWorker<Sampler>>>,
        mut generator: ResMut<ChunkGenerator<Sampler>>,
        material: Res<ChunkMaterial<Sampler, Material>>,
    ) {
        if !compute_worker.ready() {
            return;
        };

        let Some(chunk_position) = generator.current_chunk else {
            return;
        };

        // info!("Finished chunk: {chunk_position:?}");

        let vertices =
            Self::read_vec::<Vertex>(&compute_worker, "out_vertices", "out_vertices_len");
        let triangles =
            Self::read_vec::<Triangle>(&compute_worker, "out_triangles", "out_triangles_len");

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
            Transform::from_translation(generator.chunk_to_position(chunk_position)),
            Chunk::<Sampler>(chunk_position, std::marker::PhantomData::<Sampler>),
        ));

        generator.current_chunk = None;
        generator
            .loaded_chunks
            .insert(chunk_position, LoadState::Finished);
    }

    fn read_vec<T: Pod + Zeroable>(
        worker: &AppComputeWorker<MarchingCubesComputeWorker<Sampler>>,
        name: &str,
        len_name: &str,
    ) -> Vec<T> {
        let len = worker.read::<u32>(len_name) as usize;
        let size = std::mem::size_of::<T>();
        let bytes = worker.read_raw(name);
        bytemuck::cast_slice::<u8, T>(&bytes[..len * size]).to_vec()
    }

    fn start_chunks(
        mut compute_worker: ResMut<AppComputeWorker<MarchingCubesComputeWorker<Sampler>>>,
        mut generator: ResMut<ChunkGenerator<Sampler>>,
    ) {
        if generator.current_chunk.is_some() {
            return;
        }

        let Some(chunk_position) = generator.chunks_to_load.pop() else {
            return;
        };

        generator.current_chunk = Some(chunk_position);

        compute_worker.write("chunk_position", &chunk_position);
        let empty_densities = vec![0.0f32; generator.num_samples_per_axis().pow(3) as usize];
        compute_worker.write_slice("densities", empty_densities.as_slice());
        compute_worker.write("out_vertices_len", &0u32);
        compute_worker.write("out_triangles_len", &0u32);
        compute_worker.execute();
    }
}

impl<Sampler: ComputeShader + Send + Sync + 'static, Material: Asset + bevy::prelude::Material>
    Plugin for MarchingCubesPlugin<Sampler, Material>
{
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<AppComputePlugin>() {
            app.add_plugins(AppComputePlugin);
        }

        app.add_plugins(AppComputeWorkerPlugin::<MarchingCubesComputeWorker<Sampler>>::default())
            .add_systems(
                Update,
                (
                    Self::update_chunk_loaders,
                    Self::finish_chunks,
                    Self::queue_chunks,
                    Self::start_chunks,
                )
                    .chain()
                    .in_set(ChunkGenSystems),
            );
    }
}

#[derive(SystemSet, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ChunkGenSystems;

const WORKGROUP_SIZE: u32 = 8;

#[derive(TypePath)]
struct MarchingCubesShader;

impl ComputeShader for MarchingCubesShader {
    fn shader() -> ShaderRef {
        "marching_cubes.wgsl".into()
    }
}

#[derive(Resource)]
struct MarchingCubesComputeWorker<T> {
    _marker: std::marker::PhantomData<T>,
}

impl<T: ComputeShader + Send + Sync + 'static> ComputeWorker for MarchingCubesComputeWorker<T> {
    fn build(world: &mut World) -> AppComputeWorker<Self> {
        let Some(generator) = world.get_resource::<ChunkGenerator<T>>() else {
            panic!(
                "ChunkGenerator<{}> resource not found",
                std::any::type_name::<T>()
            )
        };
        let num_voxels_per_axis = generator.num_voxels_per_axis;
        let num_samples_per_axis = generator.num_samples_per_axis();
        let chunk_size = generator.chunk_size;
        let surface_threshold = generator.surface_threshold;
        let max_num_vertices = generator.max_num_vertices();
        let max_num_triangles = generator.max_num_triangles();
        let sampler_dispatch_size =
            (num_samples_per_axis as f32 / WORKGROUP_SIZE as f32).ceil() as u32;
        let marching_cubes_dispatch_size =
            (num_voxels_per_axis as f32 / WORKGROUP_SIZE as f32).ceil() as u32;

        AppComputeWorkerBuilder::new(world)
            .add_uniform("chunk_position", &IVec3::ZERO)
            .add_empty_rw_storage(
                "densities",
                size_of::<f32>() as u64 * num_samples_per_axis.pow(3) as u64,
            )
            .add_uniform("num_voxels_per_axis", &num_voxels_per_axis)
            .add_uniform("num_samples_per_axis", &num_samples_per_axis)
            .add_uniform("chunk_size", &chunk_size)
            .add_uniform("surface_threshold", &surface_threshold)
            .add_empty_staging(
                "out_vertices",
                size_of::<Vertex>() as u64 * max_num_vertices,
            )
            .add_empty_staging("out_vertices_len", size_of::<u32>() as u64)
            .add_empty_staging(
                "out_triangles",
                size_of::<Triangle>() as u64 * max_num_triangles,
            )
            .add_empty_staging("out_triangles_len", size_of::<u32>() as u64)
            .add_pass::<T>(
                [
                    sampler_dispatch_size,
                    sampler_dispatch_size,
                    sampler_dispatch_size,
                ],
                &[
                    "chunk_position",
                    "num_voxels_per_axis",
                    "num_samples_per_axis",
                    "chunk_size",
                    "densities",
                ],
            )
            .add_pass::<MarchingCubesShader>(
                [
                    marching_cubes_dispatch_size,
                    marching_cubes_dispatch_size,
                    marching_cubes_dispatch_size,
                ],
                &[
                    "densities",
                    "num_voxels_per_axis",
                    "num_samples_per_axis",
                    "chunk_size",
                    "surface_threshold",
                    "out_vertices",
                    "out_vertices_len",
                    "out_triangles",
                    "out_triangles_len",
                ],
            )
            .asynchronous(Some(Duration::from_millis(1000)))
            .one_shot()
            .build()
    }
}

#[derive(Component, Default, Debug)]
pub struct ChunkLoader<T> {
    pub position: IVec3,
    pub loading_radius: i32,
    _marker: std::marker::PhantomData<T>,
}

impl<T> ChunkLoader<T> {
    pub fn new(loading_radius: i32) -> Self {
        Self {
            position: IVec3::ZERO,
            loading_radius,
            _marker: std::marker::PhantomData,
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
