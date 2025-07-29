// Most of this code is from https://github.com/SebLague/Terraforming

use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};

use crate::compute::{do_the_compute_thing, MarchingCubesBuffers, MarchingCubesPipelines};
use crate::terrain_sampler::DensitySampler;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct Vertex {
    pub position: Vec3,
    _padding1: f32,
    pub normal: Vec3,
    _padding2: f32,
}

#[derive(Debug, Clone)]
pub struct Triangle {
    pub vertex_a: u32,
    pub vertex_b: u32,
    pub vertex_c: u32,
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
        self.num_voxels_per_axis + 1
    }

    pub fn max_num_vertices(&self) -> u64 {
        self.max_num_triangles() * 3
    }

    pub fn max_num_triangles(&self) -> u64 {
        (self.num_voxels_per_axis as u64).pow(3) * 5
    }
}

impl<T: Default> Default for ChunkGenerator<T> {
    fn default() -> Self {
        Self {
            surface_threshold: 0.5,
            num_voxels_per_axis: 32,
            chunk_size: 32.0,
            terrain_sampler: T::default(),
        }
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

    fn sample_density(&self, chunk_id: IVec3, sample_id: IVec3) -> f32 {
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

    pub fn generate_chunk(
        &self,
        chunk_id: IVec3,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        pipelines: &MarchingCubesPipelines,
        buffers: &MarchingCubesBuffers,
    ) -> Mesh {
        let mut densities = Vec::<f32>::new();
        for x in 0..self.num_samples_per_axis() {
            for y in 0..self.num_samples_per_axis() {
                for z in 0..self.num_samples_per_axis() {
                    densities.push(
                        self.sample_density(chunk_id, IVec3::new(x as i32, y as i32, z as i32)),
                    );
                }
            }
        }

        let (vertices, triangles) = do_the_compute_thing(
            render_device,
            render_queue,
            pipelines,
            buffers,
            densities.as_slice(),
            self.surface_threshold,
            self.num_voxels_per_axis,
            self.chunk_size,
            self.max_num_vertices(),
            self.max_num_triangles(),
        );

        Mesh::new(
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
        )
    }
}
