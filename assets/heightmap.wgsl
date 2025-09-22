struct Vertex {
	height: f32,
	normal: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> heightmap_sampler: sampler;

@group(0) @binding(1)
var<uniform> heightmap: texture_2d<f32>;

@group(0) @binding(2)
var<uniform> num_squares_per_axis: u32;

@group(0) @binding(3)
var<uniform> num_chunks_per_world_axis: u32;

@group(0) @binding(4)
var<storage, read_write> vertices: array<Vertex>;

@group(0) @binding(5)
var<uniform> vertices_len: u32;

fn coord_to_tex(coord: vec2<i32>) -> vec3<f32> {
	return vec2<f32>(coord) / vec2<f32>(num_squares_per_axis) / vec2<f32>(num_chunks_per_world_axis);
}

fn sample_height(coord: vec2<i32>) -> f32 {
	return textureSample(heightmap, heightmap_sampler, coord_to_tex(coord));
}

fn calculate_normal(coord: vec2<i32>) -> vec3<f32> {
	let offset_x = vec2<i32>(1, 0);
	let offset_y = vec2<i32>(0, 1);

	let left = vec3<i32>(coord - offset_x, sample_height(coord - offset_x));
	let right = vec3<i32>(coord + offset_x, sample_height(coord + offset_x));
	let up = vec3<i32>(coord + offset_y, sample_height(coord + offset_y));
	let down = vec3<i32>(coord - offset_y, sample_height(coord - offset_y));

	return normalize(cross(right - left, up - down));
}

@compute @workgroup_size(8, 8, 1)
fn main(
	@builtin(global_invocation_id) coord: vec3<u32>
) {
	if coord.x >= num_squares_per_axis || coord.y >= num_squares_per_axis {
		return;
	}

	vertices[coord.x * width + coord.y] = Vertex(sample_height(coord.xy), calculate_normal(coord.xyz));
}