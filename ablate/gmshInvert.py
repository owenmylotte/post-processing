import os
import math
import gmsh
import argparse


def load_stp_and_mesh(filename, num_elements):
    gmsh.initialize()

    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 1)  # MeshAdapt
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)  # standard
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)

    gmsh.model.add("stp")
    gmsh.merge(filename)

    gmsh.model.mesh.setSize(gmsh.model.getEntities(), num_elements)
    gmsh.model.mesh.setRecombine(3, -1)  # Recombine for all volumes
    gmsh.model.mesh.generate(3)

    gmsh.write(os.path.splitext(filename)[0] + '.msh')

    gmsh.finalize()


def invert_mesh(filename, chunk_size):
    gmsh.initialize()

    gmsh.option.setNumber("General.Verbosity", 0)

    gmsh.model.add("mesh")
    gmsh.merge(filename)

    all_entities = gmsh.model.getEntities()
    for dim, tag in all_entities:
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim, tag)
        for element_type, elements in zip(element_types, element_tags):
            node_tags, local_coord, _ = gmsh.model.mesh.getNodesByElementType(element_type, tag=tag)
            local_coord_chunked = [local_coord[i:i + 3 * chunk_size] for i in
                                   range(0, len(local_coord), 3 * chunk_size)]

            for local_coord_chunk in local_coord_chunked:
                jacobians, determinants, _ = gmsh.model.mesh.getJacobians(element_type, local_coord_chunk, tag=tag)
                for i in range(len(jacobians)):
                    if jacobians[i] < 0:
                        gmsh.model.mesh.reverse([(dim, tag)])

    gmsh.write(filename)

    gmsh.finalize()


def spherical_shell():
    gmsh.initialize()

    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)

    gmsh.model.add('spherical_shell')

    r1 = 1.0
    r2 = 2.0

    outer_sphere = gmsh.model.occ.addSphere(0, 0, 0, r2)
    inner_sphere = gmsh.model.occ.addSphere(0, 0, 0, r1)

    gmsh.model.occ.synchronize()

    # Cut the inner sphere from the outer sphere to create the shell
    shell_volume, _ = gmsh.model.occ.cut([(3, outer_sphere)], [(3, inner_sphere)])

    gmsh.model.occ.synchronize()

    lc = 0.1
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    outer_surface = gmsh.model.addPhysicalGroup(2, [outer_sphere])
    gmsh.model.setPhysicalName(2, outer_surface, "R2")

    inner_surface = gmsh.model.addPhysicalGroup(2, [inner_sphere])
    gmsh.model.setPhysicalName(2, inner_surface, "R1")

    gmsh.model.mesh.generate(3)

    gmsh.write('spherical_shell.msh')

    gmsh.finalize()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Invert negative Jacobian elements in a mesh file.')
    args = parser.parse_args()

    # Initialize the Gmsh Python API
    gmsh.initialize()

    # Create a new model
    gmsh.model.add("new_model")

    # Define the radii of the spheres
    outer_radius = 10
    inner_radius = 5
    cell_size = 0.5
    transformation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Create the spheres
    R2_outer = gmsh.model.occ.addSphere(0, 0, 0, outer_radius + cell_size, 1)
    R2_inner = gmsh.model.occ.addSphere(0, 0, 0, outer_radius, 2)

    # R1_inner = gmsh.model.occ.addSphere(0, 0, 0, inner_radius - cell_size, 8)
    R1_outer = gmsh.model.occ.addSphere(0, 0, 0, inner_radius, 3)

    shell_inner = gmsh.model.occ.addSphere(0, 0, 0, inner_radius, 4)
    shell_outer = gmsh.model.occ.addSphere(0, 0, 0, outer_radius, 5)

    # Synchronize before performing boolean operations
    gmsh.model.occ.synchronize()

    gmsh.model.occ.fragment([(3, 2)], [(3, i) for i in [1, 3, 4, 5]])

    # Create the shells representing the outer and inner boundaries
    # R2, _ = gmsh.model.occ.cut([(3, R2_inner)], [(3, R2_outer)])
    # R1, _ = gmsh.model.occ.cut([(3, R1_outer)], [(3, R1_inner)])
    # shell, _ = gmsh.model.occ.cut([(3, shell_outer)], [(3, shell_inner)])

    # Synchronize after performing boolean operations
    gmsh.model.occ.synchronize()

    # Set the characteristic length (mesh size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cell_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cell_size)

    # Create physical groups for shells and volume
    # gmsh.model.mesh.setPeriodic(2, [R2], [shell], transformation)
    # gmsh.model.mesh.setPeriodic(2, [R1], [shell], transformation)

    # Generate the 3D mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh to a file
    gmsh.write("spherical_shell.msh")

    # Finalize the Gmsh Python API
    gmsh.finalize()
