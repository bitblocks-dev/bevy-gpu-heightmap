use std::iter::once;

use bevy::prelude::*;
use bevy::render::render_resource::{BindGroupEntries, Buffer, ComputePipeline};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
};

pub struct MarchingCubesPlugin;

impl Plugin for MarchingCubesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup).add_systems(Update, rw);
    }
}

#[derive(Resource)]
struct MarchingCubesBuffers {
    buffer: Buffer,
    buffer_staging: Buffer,
}

#[derive(Resource)]
struct MarchingCubesPipelines {
    pipeline: ComputePipeline,
}

fn setup(mut commands: Commands, render_device: Res<RenderDevice>) {
    let shader_source = include_str!("../assets/marching_cubes.wgsl");
    let shader = render_device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("Marching Cubes Shader"),
        source: ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline = render_device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Marching Cubes Pipeline"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: default(),
        cache: None,
    });
    commands.insert_resource(MarchingCubesPipelines { pipeline });

    let buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Marching Cubes Buffer"),
        size: size_of::<f32>() as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buffer_staging = render_device.create_buffer(&BufferDescriptor {
        label: Some("Marching Cubes Staging Buffer"),
        size: size_of::<f32>() as u64 * 4,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    commands.insert_resource(MarchingCubesBuffers {
        buffer,
        buffer_staging,
    });
}

fn rw(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipelines: Res<MarchingCubesPipelines>,
    buffers: Res<MarchingCubesBuffers>,
    buttons: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let bind_group = render_device.create_bind_group(
        "Marching Cubes Binding",
        &pipelines.pipeline.get_bind_group_layout(0).into(),
        &BindGroupEntries::sequential((buffers.buffer.as_entire_binding(),)),
    );

    let time = time.elapsed_secs();
    let data = [time, time + 1., time + 2., time + 3.];
    let data_buf =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size_of::<f32>() * 4) };
    render_queue.write_buffer(&buffers.buffer, 0, data_buf);

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Marching Cubes Command Encoder"),
    });

    {
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&pipelines.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(10, 10, 10);
    }

    command_encoder.copy_buffer_to_buffer(
        &buffers.buffer,
        0,
        &buffers.buffer_staging,
        0,
        size_of::<f32>() as u64 * 4,
    );

    render_queue.submit(once(command_encoder.finish()));

    buffers
        .buffer_staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| {
            if let Some(err) = result.err() {
                panic!("{}", err.to_string());
            }
        });

    // change this when i inevitably want async
    render_device.poll(wgpu::MaintainBase::Wait);

    {
        let data_buf = &buffers.buffer_staging.slice(..).get_mapped_range()[..];
        let data = unsafe {
            std::slice::from_raw_parts(
                data_buf.as_ptr() as *const f32,
                data_buf.len() / size_of::<f32>(),
            )
        };
        println!("got {data:?}");
    }

    buffers.buffer_staging.unmap();
}
