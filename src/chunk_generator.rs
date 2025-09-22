use std::time::Duration;

use bevy::log::tracing_subscriber::field::debug;
use bevy::render::render_resource::binding_types::{sampler, texture_2d};
use bevy::render::render_resource::{Sampler, SamplerDescriptor, TextureView, TextureViewId};
use bevy::{platform::collections::HashMap, render::render_resource::Texture};
use bevy::prelude::*;
use bevy_app_compute::prelude::*;
use bytemuck::{Pod, Zeroable};

#[derive(Component, Debug, Clone, Copy)]
pub struct Chunk {
    pub position: IVec2,
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Vertex {
    height: f32,
    normal: Vec3,
}

#[derive(Resource, Debug, Clone)]
pub struct ChunkGeneratorSettings {
    heightmap: Handle<Image>,
    num_squares_per_axis: u32,
    num_chunks_per_world_axis: u32,
    bounds: Option<GenBounds>,
}

#[derive(Debug, Clone)]
struct GenBounds {
    min: Vec2,
    max: Vec2,
}

impl ChunkGeneratorSettings {
    pub fn new(heightmap: Handle<Image>, num_squares_per_axis: u32, num_chunks_per_world_axis: u32) -> Self {
        Self {
            heightmap,
            num_squares_per_axis,
            num_chunks_per_world_axis,
            bounds: None,
        }
    }

    pub fn with_bounds(mut self, min: Vec2, max: Vec2) -> Self {
        self.bounds = Some(GenBounds { min, max });
        self
    }

    pub fn num_samples_per_axis(&self) -> u32 {
        self.num_squares_per_axis + 3 // We sample the next chunk over too for normals
    }

    pub fn grid_size(&self) -> f32 {
        self.num_chunks_per_world_axis as f32 / self.num_squares_per_axis as f32
    }

    pub fn position_to_chunk(&self, position: Vec2) -> IVec2 {
        (position / self.num_chunks_per_world_axis as f32).floor().as_ivec2()
    }

    pub fn chunk_to_position(&self, chunk: IVec2) -> Vec2 {
        chunk.as_vec2() * self.num_chunks_per_world_axis as f32
    }

    fn is_chunk_in_bounds(&self, chunk_position: IVec2) -> bool {
        if let Some(bounds) = &self.bounds {
            let position = self.chunk_to_position(chunk_position);
            position.x >= bounds.min.x
                && position.x <= bounds.max.x
                && position.y >= bounds.min.y
                && position.y <= bounds.max.y
        } else {
            true
        }
    }
}

#[derive(Resource, Debug, Clone)]
pub struct ChunkGeneratorCache {
    loaded_chunks: HashMap<IVec2, LoadState>,
    chunks_to_load: Vec<IVec2>,
    current_chunk: Option<IVec2>,
}

impl ChunkGeneratorCache {
    pub fn is_chunk_marked(
        &self,
        settings: &ChunkGeneratorSettings,
        chunk_position: IVec2,
    ) -> bool {
        !settings.is_chunk_in_bounds(chunk_position)
            || self.loaded_chunks.contains_key(&chunk_position)
    }

    pub fn is_chunk_generated(
        &self,
        settings: &ChunkGeneratorSettings,
        chunk_position: IVec2,
    ) -> bool {
        !settings.is_chunk_in_bounds(chunk_position)
            || matches!(
                self.loaded_chunks.get(&chunk_position),
                Some(LoadState::Finished)
            )
    }

    pub fn is_chunk_with_position_marked(
        &self,
        settings: &ChunkGeneratorSettings,
        position: Vec2,
    ) -> bool {
        self.is_chunk_marked(settings, settings.position_to_chunk(position))
    }

    pub fn is_chunk_with_position_generated(
        &self,
        settings: &ChunkGeneratorSettings,
        position: Vec2,
    ) -> bool {
        self.is_chunk_generated(settings, settings.position_to_chunk(position))
    }
}

impl Default for ChunkGeneratorCache {
    fn default() -> Self {
        Self {
            loaded_chunks: default(),
            chunks_to_load: default(),
            current_chunk: default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LoadState {
    Loading,
    Finished,
}

pub struct SampleContext<'a> {
    pub world_position: Vec2,
    pub local_position: Vec2,
    pub settings: &'a ChunkGeneratorSettings,
}
pub struct HeightmapPlugin<Material> {
    _marker: std::marker::PhantomData<Material>,
}

impl<Material> Default for HeightmapPlugin<Material> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<
        Material: Asset + bevy::prelude::Material,
    > HeightmapPlugin<Material>
{
    fn update_chunk_loaders(
        settings: Res<ChunkGeneratorSettings>,
        mut chunk_loaders: Query<
            (&mut ChunkLoader, &GlobalTransform),
            Changed<GlobalTransform>,
        >,
    ) {
        for (mut chunk_loader, transform) in chunk_loaders.iter_mut() {
            let chunk_position = (transform.translation().xy() / settings.num_chunks_per_world_axis as f32)
                .floor()
                .as_ivec2();

            // Properly update change detection
            if chunk_loader.position != chunk_position {
                chunk_loader.position = chunk_position;
            }
        }
    }

    fn queue_chunks(
        settings: Res<ChunkGeneratorSettings>,
        mut cache: ResMut<ChunkGeneratorCache>,
        chunk_loaders: Query<&ChunkLoader, Changed<ChunkLoader>>,
    ) {
        for chunk_loader in chunk_loaders.iter() {
            let mut load_order = Vec::new();
            for x in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
                for y in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
                    load_order.push(Vec2::new(x as f32, y as f32));
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
                let chunk_position = chunk_loader.position + offset.as_ivec2();
                if !cache.is_chunk_marked(&settings, chunk_position) {
                    cache
                        .loaded_chunks
                        .insert(chunk_position, LoadState::Loading);
                    cache.chunks_to_load.push(chunk_position);
                    // info!("Queued chunk for loading: {chunk_position:?}");
                }
            }
        }
    }

    fn finish_chunks(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        compute_worker: Res<AppComputeWorker<HeightmapComputeWorker>>,
        settings: Res<ChunkGeneratorSettings>,
        mut cache: ResMut<ChunkGeneratorCache>,
        material: Res<ChunkMaterial<Material>>,
    ) {
        if !compute_worker.ready() {
            return;
        };

        let Some(chunk_position) = cache.current_chunk else {
            return;
        };

        // info!("Finished chunk: {chunk_position:?}");

        let vertices =
            Self::read_vec::<Vertex>(&compute_worker, "vertices", "vertices_len");

        if !vertices.is_empty() {
            let mesh = Mesh::new(
                bevy::render::mesh::PrimitiveTopology::TriangleList,
                bevy::render::render_asset::RenderAssetUsages::RENDER_WORLD,
            )
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_POSITION,
                vertices.iter().map(|v| v.height).collect::<Vec<_>>(),
            )
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_NORMAL,
                vertices.iter().map(|v| v.normal).collect::<Vec<_>>(),
            )
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_UV_0,
                vertices.iter().map(|v| v.height).collect::<Vec<_>>(),
            );
            commands.spawn((
                Name::new(format!("Chunk {chunk_position:?}")),
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(material.material.clone()),
                Transform::from_translation(
                    (settings.chunk_to_position(chunk_position)
                        + Vec2::splat(settings.grid_size() / 2.0))
                        .extend(0.0),
                ),
                Chunk { position: chunk_position },
            ));
        }

        cache.current_chunk = None;
        cache
            .loaded_chunks
            .insert(chunk_position, LoadState::Finished);
    }

    fn read_vec<T: Pod + Zeroable>(
        worker: &AppComputeWorker<HeightmapComputeWorker>,
        name: &str,
        len_name: &str,
    ) -> Vec<T> {
        let len = worker.read::<u32>(len_name) as usize;
        let size = std::mem::size_of::<T>();
        let bytes = worker.read_raw(name);
        bytemuck::cast_slice::<u8, T>(&bytes[..len * size]).to_vec()
    }

    fn start_chunks(
        mut compute_worker: ResMut<AppComputeWorker<HeightmapComputeWorker>>,
        settings: Res<ChunkGeneratorSettings>,
        mut cache: ResMut<ChunkGeneratorCache>,
    ) {
        if cache.current_chunk.is_some() {
            return;
        }

        let Some(chunk_position) = cache.chunks_to_load.pop() else {
            return;
        };

        cache.current_chunk = Some(chunk_position);

        let empty_densities = vec![0.0f32; settings.num_samples_per_axis().pow(4) as usize];
        compute_worker.write_slice("vertices", empty_densities.as_slice());
        compute_worker.write("vertices_len", &0u32);
        compute_worker.execute();
    }
}

impl<
        Material: Asset + bevy::prelude::Material,
    > Plugin for HeightmapPlugin<Material>
{
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<AppComputePlugin>() {
            app.add_plugins(AppComputePlugin);
        }

        app.add_plugins(AppComputeWorkerPlugin::<HeightmapComputeWorker>::default())
            .add_systems(
                Update,
                (
                    Self::update_chunk_loaders,
                    Self::finish_chunks,
                    Self::queue_chunks,
                    Self::start_chunks,
                )
                    .chain()
                    .in_set(ChunkGenSystems)
                    .run_if(resource_exists::<ChunkGeneratorCache>),
            )
            .init_resource::<ChunkGeneratorCache>();
    }
}

#[derive(SystemSet, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ChunkGenSystems;

const WORKGROUP_SIZE: u32 = 8;

pub trait ChunkComputeShader: ComputeShader {
    fn build_worker_extra<W: ComputeWorker>(_compute_worker: &mut AppComputeWorkerBuilder<W>) {
        // Default implementation does nothing
    }
    fn extra_sample_bindings() -> Vec<&'static str> {
        // Default implementation returns an empty vector
        Vec::new()
    }
}

#[derive(TypePath)]
struct HeightmapShader;

impl ComputeShader for HeightmapShader {
    fn shader() -> ShaderRef {
        "heightmap.wgsl".into()
    }
}

pub type ChunkComputeWorker = AppComputeWorker<HeightmapComputeWorker>;

#[derive(Resource)]
pub struct HeightmapComputeWorker;

impl ComputeWorker
    for HeightmapComputeWorker
{
    fn build(world: &mut World) -> AppComputeWorker<Self> {
        // Extract needed values before mutably borrowing world
        let (num_squares_per_axis, num_chunks_per_world_axis, heightmap_handle) = {
            let generator = world
                .get_resource::<ChunkGeneratorSettings>()
                .expect("ChunkGeneratorSettings resource not found, did you remember to add it?");
            (
                generator.num_squares_per_axis,
                generator.num_chunks_per_world_axis,
                generator.heightmap.clone(),
            )
        };

        // Extract heightmap before mutable borrow
        let heightmap = {
            let images = world
                .get_resource::<Assets<Image>>()
                .expect("Assets<Image> resource not found");
            images
                .get(&heightmap_handle)
                .expect("Heightmap handle not found in Assets<Image>")
                .clone()
        };

        let heightmap_dispatch_size =
            (num_squares_per_axis as f32 / WORKGROUP_SIZE as f32).ceil() as u32;

        // All immutable borrows are dropped here, so we can mutably borrow world
        let mut worker = AppComputeWorkerBuilder::new(world);
        worker
            .add_empty_rw_storage(
                "vertices",
                size_of::<f32>() as u64 * num_squares_per_axis.pow(2) as u64,
            )
            .add_uniform("num_squares_per_axis", &num_squares_per_axis)
            .add_uniform("num_chunks_per_world_axis", &num_chunks_per_world_axis)
            .add_texture_view(&heightmap)
            .add_pass::<HeightmapShader>(
                [
                    heightmap_dispatch_size,
                    heightmap_dispatch_size,
                    1,
                ],
                &[
                    "heightmap_sampler",
                    "heightmap",
                    "num_squares_per_axis",
                    "num_samples_per_axis",
                    "num_chunks_per_world_axis",
                    "vertices",
                    "vertices_len",
                ],
            )
            .asynchronous(Some(Duration::from_millis(1000)))
            .one_shot();

        worker.build()
    }
}

#[derive(Component, Default, Debug)]
pub struct ChunkLoader {
    pub position: IVec2,
    pub loading_radius: i32,
}

impl ChunkLoader {
    pub fn new(loading_radius: i32) -> Self {
        Self {
            position: IVec2::ZERO,
            loading_radius,
        }
    }
}

#[derive(Resource, Debug)]
pub struct ChunkMaterial<Material: Asset> {
    pub material: Handle<Material>,
}

impl<Material: Asset> ChunkMaterial<Material> {
    pub fn new(material: Handle<Material>) -> Self {
        Self {
            material,
        }
    }
}
