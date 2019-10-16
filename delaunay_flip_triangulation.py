import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def get_index_to_start_hull(points):
    min_y = np.argmin(points[:,1])
    index_to_use = min_y
    index_elements_to_consider = []

    for i, element in enumerate(points):
        if element[1] == points[min_y, 1]:
            index_elements_to_consider.append(i)

    max_sum = 0
    for index in index_elements_to_consider:
        current_sum = abs(points[index, 0] - points[index, 1])

        if current_sum > max_sum:
            index_to_use = index
            max_sum = current_sum

    return index_to_use

def get_angle_of_points(points, index_to_start_hull):
    array_angles = np.zeros(len(points))

    start_point = points[index_to_start_hull]

    for i, current_point in enumerate(points):
        if i == index_to_start_hull:
            array_angles[i] = 0
            continue
        
        diff_point = (current_point - start_point)
        p = (diff_point) / np.sqrt(np.dot(diff_point, diff_point))
        array_angles[i] = np.arccos(np.dot(np.array([1, 0]), p))

    return array_angles

def convex_angle(point_1, point_2, point_3):
    crossp = ((point_2[0] - point_1[0]) * (point_3[1] - point_1[1]) -
              (point_2[1] - point_1[1]) * (point_3[0] - point_1[0]))
    return crossp > 0

# point_1 = [421, 137]
# point_2 = [360, 111]
# point_3 = [510, 166]
# result = convex_angle(point_1, point_2, point_3)
# print(result)

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

# a = np.array([1, 0.7])
# b = np.array([0, 0])
# c = np.array([1, 0])
# angle = calculate_angle(a, b, c)
# print(angle)

def flip(triangle_1, triangle_2):
    vertex_list = [triangle_1[0], triangle_1[1], triangle_1[2], triangle_2[0], triangle_2[1], triangle_2[2]]
    repeated_vertexes = list(set([x for x in vertex_list if vertex_list.count(x) > 1]))

    new_edge = []
    for vertex in vertex_list:
        repeated = False
        for repeated_vertex in repeated_vertexes:
            if vertex == repeated_vertex:
                repeated = True
                break
        if not repeated:
            new_edge.append(vertex)

    new_triangle_1 = [repeated_vertexes[0], new_edge[0], new_edge[1]]
    new_triangle_2 = [repeated_vertexes[1], new_edge[0], new_edge[1]]
    new_triangle_1.sort()
    new_triangle_2.sort()

    return new_triangle_1, new_triangle_2

# triangle_1 = [25, 13, 20]
# triangle_2 = [13, 26, 20]
# n_t1, n_t2 = flip(triangle_1, triangle_2)

def convex_polygon(triangle_1, triangle_2, points):
    vertex_to_verify = []
    triangles = [triangle_1, triangle_2]
    for triangle in triangles:
        for i in range(3):
            vertex_to_verify.append(triangle[i])

    vertex_to_verify = list(set(vertex_to_verify))
    vertex_to_verify.sort()

    is_convex_polygon = True
    for i in range(len(vertex_to_verify)):
        if not convex_angle(points[vertex_to_verify[i - 2]], points[vertex_to_verify[i - 1]], points[vertex_to_verify[i]]):
            is_convex_polygon = False
            break

    return is_convex_polygon

    # edges_to_use = []
    # vertex_of_triangles = []
    # points_to_verify_convexity = []
    # triangles = [triangle_1, triangle_2]
    # combination_edges = [[0,1], [0, 2], [1, 2]]

    # # Take all edges of triangles
    # for triangle in triangles:
    #     for edge in combination_edges:
    #         edges_to_use.append([triangle[edge[0]], triangle[edge[1]]])
    #     for i in range(3):
    #         vertex_of_triangles.append(triangle[i])

    # # Delete commom edges
    # edges_to_delete = []
    # for i in range(len(edges_to_use)):
    #     if all(elem in edges_to_use[i] for elem in commom_edge):
    #         edges_to_delete.append(i)

    # edges_to_delete.sort(reverse=True)
    # for index_edge_to_delete in edges_to_delete:
    #     edges_to_use.pop(index_edge_to_delete)

    # # Delete duplicated vertexes
    # # vertex_of_triangles = list(set(vertex_of_triangles))
    # vertex_of_triangles = list(dict.fromkeys(vertex_of_triangles))

    # for vertex in vertex_of_triangles:
    #     points_to_verify = []
    #     edges_repeated_elements = []
    #     for edge in edges_to_use:
    #         if vertex in edge:
    #             edges_repeated_elements.append(edge)
    #     for edge in edges_repeated_elements:
    #         points_to_verify.append(edge[0])
    #         points_to_verify.append(edge[1])

    #     # From 4 points take 3 not repeated and leave the repeated in the middle of the other two
    #     points_to_verify.sort()
    #     repeated_point = None
    #     for i in range(1, len(points_to_verify)):
    #         if points_to_verify[i - 1] == points_to_verify[i]:
    #             repeated_point = points_to_verify[i]
    #             break
    #     points_in_order = []
    #     for point in points_to_verify:
    #         if point != repeated_point:
    #             points_in_order.append(point)
        
    #     points_in_order.insert(1, repeated_point)
    #     # points_in_order.sort()
    #     points_to_verify_convexity.append(points_in_order)
    
    # is_convex_polygon = True
    # for index_points in points_to_verify_convexity:
    #     if not convex_angle(points[index_points[0]], points[index_points[1]], points[index_points[2]]):
    #         is_convex_polygon = False
    #         break

    # return is_convex_polygon

# # Chamada convex_polygon
# triangle_1 = [25, 13, 20]
# triangle_2 = [13, 26, 20]
# points = np.loadtxt("dataset/fecho1.txt").astype(np.float)
# convex_polygon(triangle_1, triangle_2, points)

def create_data_structure(triangulation_indexes):
    data_structure = []

    for k, indexes in enumerate(triangulation_indexes):
        triangle_structure = [indexes[0], indexes[1], indexes[2], None, None, None]

        edge_triangle_0 = [indexes[1], indexes[2]]
        edge_triangle_1 = [indexes[2], indexes[0]]
        edge_triangle_2 = [indexes[0], indexes[1]]

        edges_to_verify = [edge_triangle_0, edge_triangle_1, edge_triangle_2]
        
        for i in range(len(edges_to_verify)):
            for j in range(len(triangulation_indexes)):
                if k == j:
                    continue
                if all(elem in triangulation_indexes[j] for elem in edges_to_verify[i]):
                    triangle_structure[i + 3] = j

        data_structure.append(triangle_structure)

    return data_structure

def update_data_structure(new_triangle_1, new_triangle_2, index_1, index_2, triangulation):
    triangles_to_update = [new_triangle_1, new_triangle_2]
    index_triangles_to_update = [index_1, index_2]

    for i, index_triangle in enumerate(index_triangles_to_update):
        index_neibourhood_to_update = []
        for j, vertex in enumerate(triangles_to_update[i]):
            triangulation[index_triangle][j] = vertex
            index_neibourhood_to_update.append(triangulation[index_triangle][j + 3])

    for index_triangle in index_neibourhood_to_update:
        if index_triangle not in index_triangles_to_update:
            elements_to_analyse_idxs = range(int(len(triangulation[index_triangle]) / 2), len(triangulation[index_triangle]))

            for j, element_to_analyse_idx in enumerate(elements_to_analyse_idxs):
                for k, index_triangle_to_update in enumerate(index_triangles_to_update):
                    if triangulation[index_triangle][element_to_analyse_idx] == index_triangle_to_update:
                        vertexes_set_adjacent_triangle = deepcopy(triangulation[index_triangle][:int(len(triangulation[index_triangle]) / 2)])
                        del vertexes_set_adjacent_triangle[j]

                        vertexes_set_new_triangle = triangulation[index_triangle_to_update][:int(len(triangulation[index_triangle_to_update]) / 2)]
                        
                        vertexes_set_of_adjacent_triangle_in_new_triangle = []
                        for vertex_point_adjacent_triangle in vertexes_set_adjacent_triangle:
                            if vertex_point_adjacent_triangle in vertexes_set_new_triangle:
                                vertexes_set_of_adjacent_triangle_in_new_triangle.append(vertex_point_adjacent_triangle)

                        if len(vertexes_set_of_adjacent_triangle_in_new_triangle) != len(vertexes_set_adjacent_triangle):
                            triangulation[index_triangle][element_to_analyse_idx] = index_triangles_to_update[k - 1]
                        
                        break
        else:
            edge_triangle_0 = [triangulation[index_triangle][1], triangulation[index_triangle][2]]
            edge_triangle_1 = [triangulation[index_triangle][2], triangulation[index_triangle][0]]
            edge_triangle_2 = [triangulation[index_triangle][0], triangulation[index_triangle][1]]

            edges_to_verify = [edge_triangle_0, edge_triangle_1, edge_triangle_2]

            for l in range(len(edges_to_verify)):
                for triangle_to_update in index_neibourhood_to_update:
                    if index_triangle == triangle_to_update:
                        continue
                    triangle = [triangulation[triangle_to_update][0], triangulation[triangle_to_update][1], triangulation[triangle_to_update][2]]
                    if all(elem in triangle for elem in edges_to_verify[l]):
                        triangulation[index_triangle][l + 3] = triangle_to_update

    return triangulation

def get_angle_polygon(triangle_1, triangle_2, points):
    triangles = [triangle_1, triangle_2]
    angle_list = []

    for triangle in triangles:
        for i, vertex in enumerate(triangle):
            # TODO Verificar se sempre tenho um ponto diferente no meio
            angle = calculate_angle(points[vertex[i]], points[vertex[(i + 1) % 3]], points[vertex[(i + 2) % 3]])
            angle_list.append(angle)

    return angle_list.sort()

def fatter_triangulation(angles_before_flip, angles_after_flip):
    fatter = False

    for k in range(len(angles_before_flip)):
        if angles_after_flip[k] > angles_before_flip[k]:
            fatter = True
            break

    return fatter

def delaunay_triangulation(triangulation, points):
    for i, triangle in enumerate(triangulation):
        for j in range(3, 6):
            if triangle[j] is not None and triangle[j] > i:
                triangle_1 = [triangle[0], triangle[1], triangle[2]]
                triangle_2 = [triangulation[triangle[j]][0], triangulation[triangle[j]][1], triangulation[triangle[j]][2]]
                if convex_polygon(triangle_1, triangle_2, points):
                    angles_before_flip = get_angle_polygon(triangle_1, triangle_2, points)
                    new_triangle_1, new_triangle_2 = flip(triangle_1, triangle_2)
                    angles_after_flip = get_angle_polygon(new_triangle_1, new_triangle_2, points)

                    if fatter_triangulation(angles_before_flip, angles_after_flip):
                        triangulation = update_data_structure(new_triangle_1, new_triangle_2, i, triangle[j], triangulation)

    return triangulation

def graham_scan(points, original_indexes):
    hull = [original_indexes[0], original_indexes[1]]
    triangulation = []

    for i in range(2, len(points)):
        if convex_angle(points[hull[-2]], points[hull[-1]], points[original_indexes[i]]):
            hull.append(original_indexes[i])
            
            # Verify if point that Grahan Scan started is alredy part of triangle
            indexes_to_consider = [hull[-2], hull[-1], original_indexes[i]]
            if hull[0] in indexes_to_consider:
                triangulation.append(indexes_to_consider)
            else:
                triangulation.append([hull[-2], hull[-1], hull[0]])
        else:
            # Add triangle with start point of Grahan
            triangulation.append([hull[-1], original_indexes[i], hull[0]])

            while len(hull) > 1 and not convex_angle(points[hull[-2]], points[hull[-1]], points[original_indexes[i]]):
                triangulation.append([original_indexes[i], hull[-1], hull[-2]])
                hull.pop()

            hull.append(original_indexes[i])

    return hull, triangulation

def triangulation(points):
    original_indexes = np.arange(len(points))

    # Get the index with small y. If there is a tie returns the index of the smallest y value with largest x value
    index_to_start_hull = get_index_to_start_hull(points)

    # Get list of angles of each point regarding to the point chosen to start the hull
    angle_of_points = get_angle_of_points(points, index_to_start_hull)
    
    # Sort list of indexes acording to the angles of all points regarding to the point used to start the hull
    index_to_sort_points = np.argsort(angle_of_points)
    original_indexes = original_indexes[index_to_sort_points]

    # Call function Graham Scan to find the convex hull
    convex_hull_indexes, triangulation_indexes = graham_scan(points, original_indexes)

    return convex_hull_indexes, triangulation_indexes

def print_solution(file_name, convex_hull_indexes):       
    with open(file_name, mode='w+') as fp:
        for index in convex_hull_indexes:
            print(str(index), file=fp)
        fp.close()

def draw_polygon(points, convex_hull_indexes, datasetname, triangulation_indexes):
    x_polygon = points[:,0]
    y_polygon = points[:,1]

    plt.scatter(x_polygon, y_polygon, color="black", marker='.')

    # for i in range (len(x_polygon)):
    #     plt.text(points[i, 0], points[i, 1], str(i))

    combination_points = [[0,1], [0,2], [1,2]]

    for triangle_indexes in triangulation_indexes:
        for indexes_points in combination_points:
            a = points[triangle_indexes[indexes_points[0]]]
            b = points[triangle_indexes[indexes_points[1]]]
            ab = np.array([a, b])
            plt.plot(ab[:,0], ab[:, 1], linewidth=1, color="green")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.suptitle(str(datasetname))
    plt.savefig("fig/"+ str(datasetname) +".png")
    plt.close()

def run():
    datasets = ["dataset/nuvem1.txt", "dataset/nuvem2.txt", "dataset/nuvem3.txt"]

    for dataset in datasets:
        # Dataset path
        INPUT_PATH = dataset
        # Load dataset
        points = np.loadtxt(INPUT_PATH).astype(np.float)
        # Call function to construct solution
        convex_hull_indexes, triangulation_indexes = triangulation(points)

        # Create data structures
        data_triangulation = create_data_structure(triangulation_indexes)

        delaunay_triangulation(data_triangulation, points)

        # Name of solution file
        output_file_name = 'delaunay{}.txt'.format(INPUT_PATH[-5])
        # Function to save solution file
        print_solution("solution/{}".format(output_file_name), convex_hull_indexes)
        # Dataset name
        datasetname = output_file_name.split(".")[0]
        # Draw solution
        draw_polygon(points, convex_hull_indexes, datasetname, triangulation_indexes)

run()
