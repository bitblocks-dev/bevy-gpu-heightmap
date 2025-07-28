@group(0) @binding(0)
var<storage, read_write> my_storage: array<f32>;

@compute @workgroup_size(1)
fn main(
	@builtin(workgroup_id) workgroup_id : vec3<u32>,
	@builtin(local_invocation_index) local_invocation_index: u32,
	@builtin(num_workgroups) num_workgroups: vec3<u32>
) {
	let workgroup_index =  
       workgroup_id.x +
       workgroup_id.y * num_workgroups.x +
       workgroup_id.z * num_workgroups.x * num_workgroups.y;

	// let global_invocation_index = workgroup_index * 1 + local_invocation_index;
	let global_invocation_index = workgroup_index + local_invocation_index;

	let index = global_invocation_index;
	if index < arrayLength(&my_storage) {
    	my_storage[index] = my_storage[index] + 1.0;
	}
}