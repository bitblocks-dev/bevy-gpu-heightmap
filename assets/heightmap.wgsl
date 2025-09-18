struct Vertex {
	position: vec3<f32>,
	normal: vec3<f32>,
}

struct Triangle {
	vertex_a: u32,
	vertex_b: u32,
	vertex_c: u32,
}

@group(0) @binding(0)
var<storage, read> heights: array<f32>;

@group(0) @binding(1)
var<uniform> num_squares_per_axis: u32;

@group(0) @binding(2)
var<uniform> num_samples_per_axis: u32;

@group(0) @binding(3)
var<uniform> chunk_size: f32;

@group(0) @binding(4)
var<uniform> surface_threshold: f32;

@group(0) @binding(5)
var<storage, read_write> out_vertices: array<Vertex>;

@group(0) @binding(6)
var<storage, read_write> out_vertices_len: atomic<u32>;

@group(0) @binding(7)
var<storage, read_write> out_triangles: array<Triangle>;

@group(0) @binding(8)
var<storage, read_write> out_triangles_len: atomic<u32>;

fn coord_to_world(coord: vec3<i32>) -> vec3<f32> {
	return vec3<f32>(coord) / vec3<f32>(num_squares_per_axis) * chunk_size;
}

fn sample_height(coord: vec3<i32>) -> f32 {
	let shifted_coord = coord + vec3<i32>(1, 1, 1); // Shift to account for the fact that we sample from the next chunk over too for normals
	return heights[shifted_coord.x * i32(num_samples_per_axis) * i32(num_samples_per_axis) + shifted_coord.y * 32(num_samples_per_axis) + shifted_coord.z];
}

fn calculate_normal(coord: vec3<i32>) -> vec3<f32> {
	let offset_x = vec3<i32>(1, 0, 0);
	let offset_y = vec3<i32>(0, 1, 0);
	let offset_z = vec3<i32>(0, 0, 1);

	let dx = sample_height(coord - offset_x) - sample_height(coord + offset_x);
	let dy = sample_height(coord - offset_y) - sample_height(coord + offset_y);
	let dz = sample_height(coord - offset_z) - sample_height(coord + offset_z);

	return normalize(vec3<f32>(dx, dy, dz));
}

// Calculate the position of the vertex
// The position lies somewhere along the edge defined by the two corner points.
// Where exactly along the edge is determined by the values of each corner point.
fn create_vertex(coord_a: vec3<i32>, coord_b: vec3<i32>) -> Vertex {

	let pos_a = coord_to_world(coord_a);
	let pos_b = coord_to_world(coord_b);
	let height_a = sample_height(coord_a);
	let height_b = sample_height(coord_b);

	// Interpolate between the two corner points based on the density
	let t = (surface_threshold - height_a) / (height_b - height_a);
	let position = pos_a + t * (pos_b - pos_a);

	// Normal:
	let normal_a = calculate_normal(coord_a);
	let normal_b = calculate_normal(coord_b);
	let normal = normalize(normal_a + t * (normal_b - normal_a));

	// Create vertex
	let vertex = Vertex(
		position,
		normal,
	);

	return vertex;
}

@compute @workgroup_size(8, 8, 8)
fn main2(
	@builtin(global_invocation_id) coord: vec3<u32>
) {
	let _height = heights[0];
	let _num_squares_per_axis = num_squares_per_axis;
	let _chunk_size = chunk_size;
	let _surface_threshold = surface_threshold;

	let vertex_index = atomicAdd(&out_vertices_len, 3u);
	out_vertices[vertex_index] = Vertex(
		vec3<f32>(coord),
		vec3<f32>(0.0, 1.0, 0.0),
	);
	out_vertices[vertex_index + 1] = Vertex(
		vec3<f32>(coord + vec3<u32>(1, 0, 0)),
		vec3<f32>(0.0, 1.0, 0.0),
	);
	out_vertices[vertex_index + 2] = Vertex(
		vec3<f32>(coord + vec3<u32>(0, 0, 1)),
		vec3<f32>(0.0, 1.0, 0.0),
	);

	let triangle_index = atomicAdd(&out_triangles_len, 1u);
	out_triangles[triangle_index] = Triangle(
		u32(vertex_index),
		u32(vertex_index + 1),
		u32(vertex_index + 2),
	);
}

@compute @workgroup_size(8, 8, 8)
fn main(
	@builtin(global_invocation_id) coord: vec3<u32>
) {
	if coord.x >= num_squares_per_axis || coord.y >= num_squares_per_axis || coord.z >= num_squares_per_axis {
		return;
	}

	// Calculate coordinates of each corner of the current cube
	// let corner_coords = array<vec3<i32>, 8>(
	// 	vec3<i32>(coord) + vec3<i32>(0, 0, 0),
	// 	vec3<i32>(coord) + vec3<i32>(1, 0, 0),
	// 	vec3<i32>(coord) + vec3<i32>(1, 0, 1),
	// 	vec3<i32>(coord) + vec3<i32>(0, 0, 1),
	// 	vec3<i32>(coord) + vec3<i32>(0, 1, 0),
	// 	vec3<i32>(coord) + vec3<i32>(1, 1, 0),
	// 	vec3<i32>(coord) + vec3<i32>(1, 1, 1),
	// 	vec3<i32>(coord) + vec3<i32>(0, 1, 1),
	// );
	// note to self: naga segfaults when this is accessed by a non-constant

	// Calculate unique index for each cube configuration.
	// There are 256 possible values (cube has 8 corners, so 2^8 possibilites).
	// A value of 0 means cube is entirely inside the surface; 255 entirely outside.
	// The value is used to look up the edge table, which indicates which edges of the cube the surface passes through.
	var cube_configuration = 0u;
	for (var i = 0; i < 8; i++) {
		// Think of the configuration as an 8-bit binary number (each bit represents the state of a corner point).
		// The state of each corner point is either 0: above the surface, or 1: below the surface.
		// The code below sets the corresponding bit to 1, if the point is below the surface.
		if sample_height(vec3<i32>(coord) + CORNER_COORDS[i]) < surface_threshold {
			cube_configuration |= 1u << u32(i);
		}
	}
	
	// Get array of the edges of the cube that the surface passes through.
	var edge_indices = TRIANGULATION[cube_configuration];

	// Create vertices for the edges that the surface passes through.
	var edge_vertices = array<Vertex, 12>();
	var edge_vertices_created = array<bool, 12>();
	var num_vertices = 0u;
	var num_vertices_for_triangles = 0u;
	for (var i = 0; i < 15; i++) {
		// If edge index is -1, then no further vertices exist in this configuration
		let edge_index = edge_indices[i];
		if edge_index == -1 { break; }
		num_vertices_for_triangles += 1;

		// If the edge has already been created, skip it.
		if edge_vertices_created[edge_index] { continue; }
		num_vertices += 1;

		// Get indices of the two corner points defining the edge that the surface passes through.
		let coord_a = vec3<i32>(coord) + CORNER_COORDS[CORNER_INDEX_A_FROM_EDGE[edge_index]];
		let coord_b = vec3<i32>(coord) + CORNER_COORDS[CORNER_INDEX_B_FROM_EDGE[edge_index]];
		edge_vertices[edge_index] = create_vertex(coord_a, coord_b);
		edge_vertices_created[edge_index] = true;
	}

	// Map the vertices and their indices to the output buffer.
	var current_vertex_index = atomicAdd(&out_vertices_len, num_vertices);
	for (var i = 0; i < 12; i++) {
		// If the edge has not been created, skip it.
		if !edge_vertices_created[i] { continue; }

		out_vertices[current_vertex_index] = edge_vertices[i];
		for (var j = 0; j < 15; j++) {
			if edge_indices[j] == i {
				edge_indices[j] = i32(current_vertex_index);
			}
		}
		current_vertex_index += 1;
	}

	// Create triangles for the current cube configuration
	var current_triangle_index = atomicAdd(&out_triangles_len, num_vertices_for_triangles / 3);
	for (var i = 0; i < 15; i += 3) {
		// If edge index is -1, then no further vertices exist in this configuration
		if edge_indices[i] == -1 { break; }

		// Create triangle
		out_triangles[current_triangle_index] = Triangle(
			u32(edge_indices[i]),
			u32(edge_indices[i + 1]),
			u32(edge_indices[i + 2]),
		);
		current_triangle_index += 1;
	}
}

// This array contains the coordinates of the corners indexed by CORNER_INDEX_A_FROM_EDGE and CORNER_INDEX_B_FROM_EDGE.
const CORNER_COORDS = array<vec3<i32>, 8>(
	vec3<i32>(0, 0, 0),
	vec3<i32>(1, 0, 0),
	vec3<i32>(1, 0, 1),
	vec3<i32>(0, 0, 1),
	vec3<i32>(0, 1, 0),
	vec3<i32>(1, 1, 0),
	vec3<i32>(1, 1, 1),
	vec3<i32>(0, 1, 1),
);