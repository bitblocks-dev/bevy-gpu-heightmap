use bevy::color::palettes::css;
use bevy::color::palettes::tailwind;
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy_marching_cubes::chunk_generator::ChunkGenerator;
use bevy_marching_cubes::compute::{
    MarchingCubesBuffers, MarchingCubesPipelines, MarchingCubesPlugin,
};
use bevy_marching_cubes::terrain_sampler::NoiseDensitySampler;
use fastnoise_lite::FastNoiseLite;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            WireframePlugin::default(),
            bevy_panorbit_camera::PanOrbitCameraPlugin,
            MarchingCubesPlugin::<ComputeSampler>::default(),
        ))
        .insert_resource(WireframeConfig {
            global: true,
            default_color: css::WHITE.into(),
        })
        .insert_resource(ChunkGenerator {
            surface_threshold: 0.5,
            num_voxels_per_axis: 32,
            chunk_size: 8.0,
            terrain_sampler: NoiseDensitySampler({
                let mut noise = FastNoiseLite::with_seed(1);
                noise.set_frequency(Some(0.2));
                noise
            }),
        })
        .add_systems(Startup, setup)
        .run();
}

type ComputeSampler = NoiseDensitySampler;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipelines: Res<MarchingCubesPipelines>,
    buffers: Res<MarchingCubesBuffers>,
    chunk_generator: Res<ChunkGenerator<ComputeSampler>>,
) {
    commands.spawn((
        Name::new("Camera"),
        Camera3d::default(),
        Transform::from_xyz(4.0, 6.5, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        bevy_panorbit_camera::PanOrbitCamera {
            button_orbit: MouseButton::Left,
            button_pan: MouseButton::Left,
            modifier_pan: Some(KeyCode::ShiftLeft),
            reversed_zoom: true,
            ..default()
        },
    ));

    commands.spawn((
        Name::new("Light"),
        DirectionalLight {
            illuminance: 4000.0,
            ..default()
        },
        Transform {
            rotation: Quat::from_euler(EulerRot::XYZ, -1.9, 0.8, 0.0),
            ..default()
        },
    ));

    let centering_offset = Vec3::splat(-chunk_generator.chunk_size * 0.5);

    commands.spawn((
        Name::new("MarchingCubesMesh"),
        Mesh3d(meshes.add(chunk_generator.generate_chunk(
            IVec3::new(0, 0, 0),
            &render_device,
            &render_queue,
            &pipelines,
            &buffers,
        ))),
        MeshMaterial3d(materials.add(Color::from(tailwind::EMERALD_500))),
        Transform::from_translation(centering_offset),
    ));
    commands.spawn((
        Name::new("MarchingCubesMesh2"),
        Mesh3d(meshes.add(chunk_generator.generate_chunk(
            Dir3::X.as_ivec3(),
            &render_device,
            &render_queue,
            &pipelines,
            &buffers,
        ))),
        MeshMaterial3d(materials.add(Color::from(tailwind::EMERALD_500))),
        Transform::from_translation(centering_offset + chunk_generator.chunk_size * Dir3::X),
    ));
}
