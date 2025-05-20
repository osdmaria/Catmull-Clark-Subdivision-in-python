import numpy as np
import trimesh
import polyscope as ps
from collections import defaultdict


# ----------- Subdivision Function -----------

def catmull_clark_subdivision(vertices, faces, iterations=1):
    for _ in range(iterations):
        n_vertices = len(vertices)
        n_faces = len(faces)

        # Step 1: Compute face points
        face_points = np.array([np.mean(vertices[face], axis=0) for face in faces])

        # Step 2: Compute edge points and adjacency info
        edge_dict = defaultdict(list)
        edge_to_faces = defaultdict(list)
        edge_to_vertices = {}
        for face_idx, face in enumerate(faces):
            for i in range(len(face)):
                v1, v2 = face[i], face[(i + 1) % len(face)]
                edge = tuple(sorted((v1, v2)))
                edge_dict[edge].append(face_idx)
                edge_to_faces[edge].append(face_idx)
                edge_to_vertices[edge] = (v1, v2)

        edge_points = {}
        for edge, face_indices in edge_to_faces.items():
            v1, v2 = edge_to_vertices[edge]
            if len(face_indices) == 2:
                edge_points[edge] = (
                    face_points[face_indices[0]] +
                    face_points[face_indices[1]] +
                    vertices[v1] + vertices[v2]
                ) / 4
            else:
                edge_points[edge] = (vertices[v1] + vertices[v2]) / 2

        # Step 3: Update vertex positions
        vertex_to_faces = defaultdict(list)
        vertex_to_edges = defaultdict(list)
        for face_idx, face in enumerate(faces):
            for v in face:
                vertex_to_faces[v].append(face_idx)

        for edge in edge_to_faces:
            v1, v2 = edge
            vertex_to_edges[v1].append(edge)
            vertex_to_edges[v2].append(edge)

        new_vertices = []
        for v in range(n_vertices):
            adj_faces = vertex_to_faces[v]
            adj_edges = vertex_to_edges[v]

            if len(adj_edges) == 0:
                new_v = vertices[v]
            elif any(len(edge_to_faces[tuple(sorted(e))]) == 1 for e in adj_edges):
                boundary_neighbors = []
                for e in adj_edges:
                    if len(edge_to_faces[tuple(sorted(e))]) == 1:
                        v1, v2 = e
                        other = v1 if v2 == v else v2
                        boundary_neighbors.append(other)
                if len(boundary_neighbors) == 2:
                    new_v = (vertices[v] +
                             vertices[boundary_neighbors[0]] +
                             vertices[boundary_neighbors[1]]) / 3
                else:
                    new_v = vertices[v]
            else:
                F = np.mean([face_points[i] for i in adj_faces], axis=0)
                R = np.mean([edge_points[tuple(sorted(e))] for e in adj_edges], axis=0)
                n = len(adj_faces)
                new_v = (F + 2 * R + (n - 3) * vertices[v]) / n

            new_vertices.append(new_v)

        # Step 4: Build new faces
        new_faces = []
        face_point_indices = n_vertices + np.arange(len(face_points))
        edge_point_indices = {
            edge: n_vertices + len(face_points) + i
            for i, edge in enumerate(sorted(edge_points.keys()))
        }

        for face_idx, face in enumerate(faces):
            vf = face_point_indices[face_idx]
            for i in range(len(face)):
                v = face[i]
                next_v = face[(i + 1) % len(face)]
                prev_v = face[(i - 1) % len(face)]
                e1 = tuple(sorted((v, next_v)))
                e2 = tuple(sorted((prev_v, v)))
                ve1 = edge_point_indices[e1]
                ve2 = edge_point_indices[e2]
                new_faces.append([v, ve1, vf, ve2])

        # Stack new vertices together
        all_edge_points = [edge_points[edge] for edge in sorted(edge_points.keys())]
        vertices = np.vstack([
            np.array(new_vertices),
            face_points,
            np.array(all_edge_points)
        ])
        faces = new_faces

    return vertices, faces


# ----------- Create Quad Cube (for testing) -----------

def create_quad_cube():
    cube_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float64)

    cube_faces = np.array([
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0]
    ], dtype=np.int64)

    return cube_vertices, cube_faces


# ----------- Main Pipeline -----------
def main():
    # to use a triangles based mesh
    mesh = trimesh.load('meshFiles/cowhead.obj')
    vertices = mesh.vertices
    faces = mesh.faces

    # to use cube (with quad))
    vertices, faces = create_quad_cube()

    # Apply Catmull-Clark subdivision
    final_vertices, final_faces = catmull_clark_subdivision(vertices, faces, iterations=3)


    ps.init()
    ps.register_surface_mesh("Subdivided Mesh", final_vertices, final_faces)
    ps.show()


if __name__ == "__main__":
    main()
