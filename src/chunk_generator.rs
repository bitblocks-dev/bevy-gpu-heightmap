// Most of this code is from https://github.com/SebLague/Terraforming

use bevy::math::{IVec2, IVec3, Vec3};
use bevy::platform::collections::HashMap;
use bevy::render::mesh::Mesh;

use crate::march_tables::*;
use crate::terrain_sampler::TerrainSampler;

#[derive(Debug, Clone)]
struct Vertex {
    position: Vec3,
    normal: Vec3,
    id: IVec2,
}

#[derive(Debug, Clone)]
struct Triangle {
    vertex_a: Vertex,
    vertex_b: Vertex,
    vertex_c: Vertex,
}

#[derive(Debug, Clone)]
pub struct ChunkGenerator<T> {
    pub surface_threshold: f32,
    pub num_voxels: i32,
    pub chunk_size: f32,
    pub terrain_sampler: T,
}

impl<T: Default> Default for ChunkGenerator<T> {
    fn default() -> Self {
        Self {
            surface_threshold: 0.5,
            num_voxels: 32,
            chunk_size: 32.0,
            terrain_sampler: T::default(),
        }
    }
}

impl<T: TerrainSampler> ChunkGenerator<T> {
    fn voxel_scale(&self) -> f32 {
        self.chunk_size / self.num_voxels as f32
    }

    fn sample_density(
        &self,
        chunk_id: IVec3,
        voxel_id: IVec3,
        cache: &mut HashMap<IVec3, f32>,
    ) -> f32 {
        *cache.entry(voxel_id).or_insert_with(|| {
            self.terrain_sampler
                .sample_density(self.coord_to_world(chunk_id, voxel_id))
        })
    }

    fn coord_to_local(&self, voxel_id: IVec3) -> Vec3 {
        voxel_id.as_vec3() * self.voxel_scale()
    }

    fn coord_to_world(&self, chunk_id: IVec3, voxel_id: IVec3) -> Vec3 {
        (chunk_id * self.num_voxels + voxel_id).as_vec3() * self.voxel_scale()
    }

    fn index_from_coord(&self, voxel_id: IVec3) -> usize {
        let num_samples = self.num_voxels + 1; // +1 because we need to include the last sample
        (voxel_id.z * num_samples * num_samples + voxel_id.y * num_samples + voxel_id.x) as usize
    }

    fn calculate_normal(
        &self,
        chunk_id: IVec3,
        voxel_id: IVec3,
        cache: &mut HashMap<IVec3, f32>,
    ) -> Vec3 {
        let offset_x = IVec3::new(1, 0, 0);
        let offset_y = IVec3::new(0, 1, 0);
        let offset_z = IVec3::new(0, 0, 1);

        let x1 = self.sample_density(chunk_id, voxel_id + offset_x, cache);
        let x2 = self.sample_density(chunk_id, voxel_id - offset_x, cache);
        let dx = x1 - x2;

        let y1 = self.sample_density(chunk_id, voxel_id + offset_y, cache);
        let y2 = self.sample_density(chunk_id, voxel_id - offset_y, cache);
        let dy = y1 - y2;

        let z1 = self.sample_density(chunk_id, voxel_id + offset_z, cache);
        let z2 = self.sample_density(chunk_id, voxel_id - offset_z, cache);
        let dz = z1 - z2;

        Vec3::new(dx, dy, dz).normalize()
    }

    // Calculate the position of the vertex
    // The position lies somewhere along the edge defined by the two corner points.
    // Where exactly along the edge is determined by the values of each corner point.
    fn create_vertex(
        &self,
        chunk_id: IVec3,
        voxel_a_id: IVec3,
        voxel_b_id: IVec3,
        cache: &mut HashMap<IVec3, f32>,
    ) -> Vertex {
        let pos_a = self.coord_to_local(voxel_a_id);
        let pos_b = self.coord_to_local(voxel_b_id);
        let density_a = self.sample_density(chunk_id, voxel_a_id, cache);
        let density_b = self.sample_density(chunk_id, voxel_b_id, cache);

        // Interpolate between the two corner points based on the density
        let t = (self.surface_threshold - density_a) / (density_b - density_a);
        let position = pos_a + t * (pos_b - pos_a);

        // Normal:
        let normal_a = self.calculate_normal(chunk_id, voxel_a_id, cache);
        let normal_b = self.calculate_normal(chunk_id, voxel_b_id, cache);
        let normal = (normal_a + t * (normal_b - normal_a)).normalize();

        // ID
        let index_a = self.index_from_coord(voxel_a_id);
        let index_b = self.index_from_coord(voxel_b_id);

        // Create vertex
        Vertex {
            position,
            normal,
            id: IVec2::new(index_a.min(index_b) as i32, index_a.max(index_b) as i32),
        }
    }

    fn process_cube(
        &self,
        chunk_id: IVec3,
        voxel_id: IVec3,
        cache: &mut HashMap<IVec3, f32>,
    ) -> Vec<Triangle> {
        let mut triangles = Vec::new();

        if voxel_id.x >= self.num_voxels
            || voxel_id.y >= self.num_voxels
            || voxel_id.z >= self.num_voxels
        {
            return triangles;
        }

        let coord = voxel_id;

        // Calculate coordinates of each corner of the current cube
        let corner_coords = [
            coord + IVec3::new(0, 0, 0),
            coord + IVec3::new(1, 0, 0),
            coord + IVec3::new(1, 0, 1),
            coord + IVec3::new(0, 0, 1),
            coord + IVec3::new(0, 1, 0),
            coord + IVec3::new(1, 1, 0),
            coord + IVec3::new(1, 1, 1),
            coord + IVec3::new(0, 1, 1),
        ];
        // Calculate unique index for each cube configuration.
        // There are 256 possible values (cube has 8 corners, so 2^8 possibilites).
        // A value of 0 means cube is entirely inside the surface; 255 entirely outside.
        // The value is used to look up the edge table, which indicates which edges of the cube the surface passes through.
        let mut cube_configuration = 0;
        for (i, corner_coord) in corner_coords.iter().enumerate() {
            // Think of the configuration as an 8-bit binary number (each bit represents the state of a corner point).
            // The state of each corner point is either 0: above the surface, or 1: below the surface.
            // The code below sets the corresponding bit to 1, if the point is below the surface.
            if self.sample_density(chunk_id, *corner_coord, cache) < self.surface_threshold {
                cube_configuration |= 1 << i;
            }
        }

        // Get array of the edges of the cube that the surface passes through.
        let edge_indices = TRIANGULATION[cube_configuration];

        // Create triangles for the current cube configuration
        for i in (0..16).step_by(3) {
            // If edge index is -1, then no further vertices exist in this configuration
            if edge_indices[i] == -1 {
                break;
            }

            // Get indices of the two corner points defining the edge that the surface passes through.
            // (Do this for each of the three edges we're currently looking at).
            let edge_index_a = edge_indices[i] as usize;
            let a0 = CORNER_INDEX_A_FROM_EDGE[edge_index_a];
            let a1 = CORNER_INDEX_B_FROM_EDGE[edge_index_a];

            let edge_index_b = edge_indices[i + 1] as usize;
            let b0 = CORNER_INDEX_A_FROM_EDGE[edge_index_b];
            let b1 = CORNER_INDEX_B_FROM_EDGE[edge_index_b];

            let edge_index_c = edge_indices[i + 2] as usize;
            let c0 = CORNER_INDEX_A_FROM_EDGE[edge_index_c];
            let c1 = CORNER_INDEX_B_FROM_EDGE[edge_index_c];

            // Calculate positions of each vertex.
            let vertex_a =
                self.create_vertex(chunk_id, corner_coords[a0], corner_coords[a1], cache);
            let vertex_b =
                self.create_vertex(chunk_id, corner_coords[b0], corner_coords[b1], cache);
            let vertex_c =
                self.create_vertex(chunk_id, corner_coords[c0], corner_coords[c1], cache);

            // Create triangle
            let tri = Triangle {
                vertex_a,
                vertex_b,
                vertex_c,
            };
            triangles.push(tri);
        }

        triangles
    }

    pub fn generate_chunk(&self, chunk_id: IVec3) -> Mesh {
        let mut cache = HashMap::<IVec3, f32>::new();

        let mut triangles = Vec::<Triangle>::new();
        for x in 0..self.num_voxels {
            for y in 0..self.num_voxels {
                for z in 0..self.num_voxels {
                    triangles.extend(self.process_cube(chunk_id, IVec3::new(x, y, z), &mut cache));
                }
            }
        }

        let vertex_data: Vec<Vertex> = triangles
            .iter()
            .flat_map(|t| [t.vertex_c.clone(), t.vertex_b.clone(), t.vertex_a.clone()])
            .collect();

        let mut vertex_positions: Vec<Vec3> = vec![];
        let mut vertex_normals: Vec<Vec3> = vec![];
        let mut indices: Vec<u32> = vec![];
        let mut vertex_index_map: HashMap<IVec2, u32> = HashMap::new();

        let mut vertex_index = 0;
        for data in vertex_data.iter() {
            if let Some(shared_vertex_index) = vertex_index_map.get(&data.id) {
                indices.push(*shared_vertex_index);
            } else {
                vertex_index_map.insert(data.id, vertex_index);
                vertex_positions.push(data.position);
                vertex_normals.push(-data.normal);
                indices.push(vertex_index);
                vertex_index += 1;
            }
        }

        Mesh::new(
            bevy::render::mesh::PrimitiveTopology::TriangleList,
            bevy::render::render_asset::RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_indices(bevy::render::mesh::Indices::U32(indices))
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertex_positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vertex_normals)
    }
}
