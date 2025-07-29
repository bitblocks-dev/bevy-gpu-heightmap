use std::iter::once;

use bevy::log::tracing_subscriber::field::debug;
use bevy::prelude::*;
use bevy::render::render_resource::{BindGroupEntries, Buffer, ComputePipeline};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoder, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
};

use crate::chunk_generator::{ChunkGenerator, Triangle, Vertex};

pub struct MarchingCubesPlugin<T> {
    _marker: std::marker::PhantomData<T>,
}

impl<T> Default for MarchingCubesPlugin<T> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Send + Sync + 'static> MarchingCubesPlugin<T> {
    fn setup_buffers(
        mut commands: Commands,
        render_device: Res<RenderDevice>,
        generator: Res<ChunkGenerator<T>>,
    ) {
        let densities = render_device.create_buffer(&BufferDescriptor {
            label: Some("Marching Cubes Densities Buffer"),
            size: size_of::<f32>() as u64 * generator.num_samples_per_axis().pow(3) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let num_voxels_per_axis = render_device.create_buffer(&BufferDescriptor {
            label: Some("Marching Cubes Num Voxels Buffer"),
            size: size_of::<u32>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chunk_size = render_device.create_buffer(&BufferDescriptor {
            label: Some("Marching Cubes Chunk Size Buffer"),
            size: size_of::<f32>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surface_threshold = render_device.create_buffer(&BufferDescriptor {
            label: Some("Marching Cubes Surface Threshold Buffer"),
            size: size_of::<f32>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_vertices = render_device.create_buffer(&BufferDescriptor {
            label: Some("Marching Cubes Vertices Buffer"),
            size: size_of::<Vertex>() as u64 * generator.max_num_vertices(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_vertices_out = render_device.create_buffer(&BufferDescriptor {
            label: Some("Marching Cubes Vertices Staging Buffer"),
            size: size_of::<Vertex>() as u64 * generator.max_num_vertices(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_vertices_len = render_device.create_buffer(&BufferDescriptor {
            label: Some("Marching Cubes Vertices Length Buffer"),
            size: size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_vertices_len_out = render_device.create_buffer(&BufferDescriptor {
            label: Some("Marching Cubes Vertices Length Staging Buffer"),
            size: size_of::<u32>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_triangles = render_device.create_buffer(&BufferDescriptor {
            label: Some("Marching Cubes Triangles Buffer"),
            size: size_of::<Triangle>() as u64 * generator.max_num_triangles(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_triangles_out = render_device.create_buffer(&BufferDescriptor {
            label: Some("Marching Cubes Triangles Staging Buffer"),
            size: size_of::<Triangle>() as u64 * generator.max_num_triangles(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_triangles_len = render_device.create_buffer(&BufferDescriptor {
            label: Some("Marching Cubes Triangles Length Buffer"),
            size: size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_triangles_len_out = render_device.create_buffer(&BufferDescriptor {
            label: Some("Marching Cubes Triangles Length Staging Buffer"),
            size: size_of::<u32>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        commands.insert_resource(MarchingCubesBuffers {
            densities,
            num_voxels_per_axis,
            chunk_size,
            surface_threshold,
            out_vertices,
            out_vertices_out,
            out_vertices_len,
            out_vertices_len_out,
            out_triangles,
            out_triangles_out,
            out_triangles_len,
            out_triangles_len_out,
        });
    }

    fn setup_pipelines(mut commands: Commands, render_device: Res<RenderDevice>) {
        let shader_source = include_str!("../assets/marching_cubes.wgsl");
        let shader = render_device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("Marching Cubes Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });
        let marching_cubes = render_device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Marching Cubes Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: default(),
            cache: None,
        });
        commands.insert_resource(MarchingCubesPipelines { marching_cubes });
    }
}

impl<T: Send + Sync + 'static> Plugin for MarchingCubesPlugin<T> {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, (Self::setup_buffers, Self::setup_pipelines));
    }
}

#[derive(Resource)]
pub struct MarchingCubesBuffers {
    densities: Buffer,
    num_voxels_per_axis: Buffer,
    chunk_size: Buffer,
    surface_threshold: Buffer,
    out_vertices: Buffer,
    out_vertices_out: Buffer,
    out_vertices_len: Buffer,
    out_vertices_len_out: Buffer,
    out_triangles: Buffer,
    out_triangles_out: Buffer,
    out_triangles_len: Buffer,
    out_triangles_len_out: Buffer,
}

#[derive(Resource)]
pub struct MarchingCubesPipelines {
    marching_cubes: ComputePipeline,
}

const WORKGROUP_SIZE: u32 = 8;

/// Queue up write operations to the buffer.
fn write_buffer_single<T>(render_queue: &RenderQueue, buffer: &Buffer, data: &T) {
    let data_buf =
        unsafe { std::slice::from_raw_parts(data as *const T as *const u8, size_of::<T>()) };
    render_queue.write_buffer(buffer, 0, data_buf);
}

/// Queue up write operations to the buffer.
fn write_buffer_slice<T>(render_queue: &RenderQueue, buffer: &Buffer, data: &[T]) {
    let data_buf = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    render_queue.write_buffer(buffer, 0, data_buf);
}

/// Copy data from one buffer to another.
/// Usually used to copy the output buffer to a staging buffer (the CPU-readable output buffer) for reading.
fn blit_buffer<T>(command_encoder: &mut CommandEncoder, src: &Buffer, dst: &Buffer, size: u64) {
    command_encoder.copy_buffer_to_buffer(src, 0, dst, 0, size_of::<T>() as u64 * size);
}

/// Tell the buffer to start mapping from the GPU to the CPU for reading.
/// Needs to be followed by polling the render device
/// to ensure the mapping is complete before reading the data.
fn map_buffer(buffer: &Buffer) {
    buffer.slice(..).map_async(wgpu::MapMode::Read, |result| {
        if let Some(err) = result.err() {
            panic!("{}", err.to_string());
        }
    });
}

/// Read any mapped data from the buffer.
/// Needs to be called after `map_buffer` and a poll.
/// Also unmaps the buffer after reading.
fn read_buffer_single<T: Clone>(buffer: &Buffer) -> T {
    let data = {
        let data_buf = buffer.slice(..).get_mapped_range();
        unsafe { std::slice::from_raw_parts(data_buf.as_ptr() as *const T, 1) }
    };
    buffer.unmap();
    data[0].clone()
}

/// Read any mapped data from the buffer.
/// Needs to be called after `map_buffer` and a poll.
/// Also unmaps the buffer after reading.
fn read_buffer_slice<T: Clone>(buffer: &Buffer) -> Vec<T> {
    let data = {
        let data_buf = buffer.slice(..).get_mapped_range();
        unsafe {
            std::slice::from_raw_parts(
                data_buf.as_ptr() as *const T,
                data_buf.len() / size_of::<T>(),
            )
        }
        .to_vec()
    };
    buffer.unmap();
    data
}

pub fn do_the_compute_thing(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    pipelines: &MarchingCubesPipelines,
    buffers: &MarchingCubesBuffers,
    densities: &[f32],
    surface_threshold: f32,
    num_voxels_per_axis: u32,
    chunk_size: f32,
    max_num_vertices: u64,
    max_num_triangles: u64,
) -> (Vec<Vertex>, Vec<Triangle>) {
    // Create the bind group entries for the compute shader.
    // This is where we bind the buffers to the shader.
    let bind_group = render_device.create_bind_group(
        "Marching Cubes Binding",
        &pipelines.marching_cubes.get_bind_group_layout(0).into(),
        &BindGroupEntries::sequential((
            buffers.densities.as_entire_binding(),
            buffers.num_voxels_per_axis.as_entire_binding(),
            buffers.chunk_size.as_entire_binding(),
            buffers.surface_threshold.as_entire_binding(),
            buffers.out_vertices.as_entire_binding(),
            buffers.out_vertices_len.as_entire_binding(),
            buffers.out_triangles.as_entire_binding(),
            buffers.out_triangles_len.as_entire_binding(),
        )),
    );

    // Write the data to the buffers.
    write_buffer_slice(render_queue, &buffers.densities, densities);
    write_buffer_single(
        render_queue,
        &buffers.num_voxels_per_axis,
        &num_voxels_per_axis,
    );
    write_buffer_single(render_queue, &buffers.chunk_size, &chunk_size);
    write_buffer_single(render_queue, &buffers.surface_threshold, &surface_threshold);
    write_buffer_single(render_queue, &buffers.out_vertices_len, &0u32);
    write_buffer_single(render_queue, &buffers.out_triangles_len, &0u32);

    // Create the thing that lets us make passes.
    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Marching Cubes Command Encoder"),
    });

    // Set up and queue dispatching the compute shader.
    {
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&pipelines.marching_cubes);
        pass.set_bind_group(0, &bind_group, &[]);
        let dispatch_size = (num_voxels_per_axis as f32 / WORKGROUP_SIZE as f32).ceil() as u32;
        pass.dispatch_workgroups(dispatch_size, dispatch_size, dispatch_size);
    }

    // Copy the output buffer to the other output buffer that we can actually read from.
    blit_buffer::<Vertex>(
        &mut command_encoder,
        &buffers.out_vertices,
        &buffers.out_vertices_out,
        max_num_vertices,
    );
    blit_buffer::<u32>(
        &mut command_encoder,
        &buffers.out_vertices_len,
        &buffers.out_vertices_len_out,
        1,
    );
    blit_buffer::<Triangle>(
        &mut command_encoder,
        &buffers.out_triangles,
        &buffers.out_triangles_out,
        max_num_triangles,
    );
    blit_buffer::<u32>(
        &mut command_encoder,
        &buffers.out_triangles_len,
        &buffers.out_triangles_len_out,
        1,
    );

    // Submit the whole queue to the GPU.
    render_queue.submit(once(command_encoder.finish()));

    // Map the output buffer from the GPU to the CPU for reading.
    map_buffer(&buffers.out_vertices_out);
    map_buffer(&buffers.out_vertices_len_out);
    map_buffer(&buffers.out_triangles_out);
    map_buffer(&buffers.out_triangles_len_out);

    // Poll the render device to ensure the buffer is ready for reading.
    // change this when i inevitably want async
    render_device.poll(wgpu::MaintainBase::Wait);

    // Read the data from the mapped buffer and unmap it.
    let vertices_len = read_buffer_single::<u32>(&buffers.out_vertices_len_out) as usize;
    let vertices: Vec<Vertex> = read_buffer_slice::<Vertex>(&buffers.out_vertices_out)
        .into_iter()
        .take(vertices_len)
        .collect();
    let triangles_len = read_buffer_single::<u32>(&buffers.out_triangles_len_out) as usize;
    let triangles: Vec<Triangle> = read_buffer_slice::<Triangle>(&buffers.out_triangles_out)
        .into_iter()
        .take(triangles_len)
        .collect();

    (vertices, triangles)
}
