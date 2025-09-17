use bevy::color::palettes::{css, tailwind};
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::prelude::*;
use bevy_marching_cubes::chunk_generator::{
    ChunkComputeShader, ChunkGeneratorSettings, ChunkLoader, ChunkMaterial, MarchingCubesPlugin,
};
use bevy_marching_cubes::*;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            WireframePlugin::default(),
            bevy_panorbit_camera::PanOrbitCameraPlugin,
            MarchingCubesPlugin::<MyComputeSampler, StandardMaterial>::default(),
        ))
        .insert_resource(WireframeConfig {
            global: true,
            default_color: css::WHITE.into(),
        })
        .insert_resource(ChunkGeneratorSettings::<MyComputeSampler>::new(32, 8.0))
        .add_systems(Startup, setup)
        .run();
}

#[derive(TypePath)]
struct MyComputeSampler;
impl ComputeShader for MyComputeSampler {
    fn shader() -> ShaderRef {
        "sample.wgsl".into()
    }
}
impl ChunkComputeShader for MyComputeSampler {}

fn setup(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
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

    commands.spawn(ChunkLoader::<MyComputeSampler>::new(1));

    commands.insert_resource(ChunkMaterial::<MyComputeSampler, StandardMaterial>::new(
        materials.add(Color::from(tailwind::EMERALD_500)),
    ));
}
