use bevy::color::palettes::css;
use bevy::color::palettes::tailwind;
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::prelude::*;
use bevy_inspector_egui::bevy_egui::EguiPlugin;
use bevy_marching_cubes::chunk_generator::ChunkGenerator;
use bevy_marching_cubes::terrain_sampler::NoiseTerrainSampler;
use fastnoise_lite::FastNoiseLite;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            WireframePlugin::default(),
            EguiPlugin::default(),
            bevy_inspector_egui::quick::WorldInspectorPlugin::new(),
            bevy_panorbit_camera::PanOrbitCameraPlugin,
        ))
        .insert_resource(WireframeConfig {
            global: true,
            default_color: css::WHITE.into(),
        })
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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

    let mut noise = FastNoiseLite::with_seed(1);
    noise.set_frequency(Some(0.2));

    let chunk_generator = ChunkGenerator::<NoiseTerrainSampler> {
        surface_threshold: 0.5,
        num_voxels: 32,
        chunk_size: 8.0,
        terrain_sampler: NoiseTerrainSampler(noise),
    };

    let centering_offset = Vec3::splat(-chunk_generator.chunk_size * 0.5);

    commands.spawn((
        Name::new("MarchingCubesMesh"),
        Mesh3d(meshes.add(chunk_generator.generate_chunk(IVec3::new(0, 0, 0)))),
        MeshMaterial3d(materials.add(Color::from(tailwind::EMERALD_500))),
        Transform::from_translation(centering_offset),
    ));
    commands.spawn((
        Name::new("MarchingCubesMesh2"),
        Mesh3d(meshes.add(chunk_generator.generate_chunk(Dir3::X.as_ivec3()))),
        MeshMaterial3d(materials.add(Color::from(tailwind::EMERALD_500))),
        Transform::from_translation(centering_offset + chunk_generator.chunk_size * Dir3::X),
    ));
}
