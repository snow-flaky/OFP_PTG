from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon


def sign(x1, y1, x2, y2, x3, y3):
    return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)


def PointInTriangle(x1, y1, x2, y2, x3, y3, x, y):
    d1 = sign(x, y, x1, y1, x2, y2)
    d2 = sign(x, y, x2, y2, x3, y3)
    d3 = sign(x, y, x3, y3, x1, y1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def GetOuterFace(graph, NumberOfVertices, origin_pos):
    H = graph.to_directed()
    all_cycles = list(nx.simple_cycles(H))
    all_triangles = []

    for cycle in all_cycles:
        if len(cycle) == 3:
            all_triangles.append(cycle)

    for i in all_triangles:
        i.sort()

    all_triangles = np.unique(all_triangles, axis=0)
    trianlular_faces = []

    for face in all_triangles:
        cnt = 0
        for nodeid in origin_pos:

            if (PointInTriangle(origin_pos[face[0]][0], origin_pos[face[0]][1], origin_pos[face[1]][0],
                                origin_pos[face[1]][1], origin_pos[face[2]][0],
                                origin_pos[face[2]][1], origin_pos[nodeid][0], origin_pos[nodeid][1])):
                cnt = cnt + 1
        if cnt == NumberOfVertices:
            return face

    return -1


def common_edge(a, b):
    count = 0
    for i in a:
        for j in b:
            if i == j:
                count = count + 1

    return count > 1

def outer_edges(G,numVertices,origin_pos):
    outFace = GetOuterFace(G, numVertices,origin_pos)
    outer_edge_list=set()
    for i in G.edges():
        if ((outFace[0] in i) and (outFace[1] in i)) or ((outFace[2] in i) and (outFace[1] in i)) or (
                (outFace[0] in i) and (outFace[2] in i)):
            outer_edge_list.add(i)
    return list(outer_edge_list)


def outer_faces_edge_mapping(G,facelist,numVertices,origin_pos):
    outerface=[]
    outerEdge=outer_edges(G,numVertices,origin_pos)
    outFace=GetOuterFace(G,numVertices,origin_pos)
    outer_face_edge_map={}
    for i in facelist:
        if ((outFace[0] in i) and (outFace[1] in i) ) or ((outFace[2] in i) and (outFace[1] in i) ) or ((outFace[0] in i) and( outFace[2] in i) )  :
            outerface.append(i)
            for ed in outerEdge:
                if ed[0] in i and ed[1] in i:
                    outer_face_edge_map[tuple(i)] =ed
    return  outer_face_edge_map

def outer_faces(G, facelist, numVertices, origin_pos):
    outerface = []
    outFace = GetOuterFace(G, numVertices, origin_pos)
    for i in facelist:
        if ((outFace[0] in i) and (outFace[1] in i)) or ((outFace[2] in i) and (outFace[1] in i)) or (
                (outFace[0] in i) and (outFace[2] in i)):
            outerface.append(i)
    return list(outerface)


def face_list(graph, origin_pos):
    H = graph.to_directed()
    all_cycles = list(nx.simple_cycles(H))
    all_triangles = []

    for cycle in all_cycles:
        if len(cycle) == 3:
            all_triangles.append(cycle)

    for i in all_triangles:
        i.sort()

    all_triangles = np.unique(all_triangles, axis=0)
    trianlular_faces = []

    for face in all_triangles:
        flag = False
        for nodeid in origin_pos:
            if (nodeid == face[0] or nodeid == face[1] or nodeid == face[2]):
                continue

            if (PointInTriangle(origin_pos[face[0]][0], origin_pos[face[0]][1], origin_pos[face[1]][0],
                                origin_pos[face[1]][1], origin_pos[face[2]][0],
                                origin_pos[face[2]][1], origin_pos[nodeid][0], origin_pos[nodeid][1])):
                flag = True
                break
        if flag == False:
            trianlular_faces.append(face.tolist())

    return trianlular_faces


def chordless_cycle(cycle, graph):
    for i in cycle:
        if neighbour_in_cycle_vertex(cycle, i, graph):
            return True
    return False


def neighbour_in_cycle_vertex(cycle, vertex, graph):
    vertex_neighbor = graph.neighbors(vertex)
    count = 0
    for i in vertex_neighbor:
        if i in cycle:
            count = count + 1
    if count > 2:
        return True
    return False


def neighbour_in_cycle_vertex_2(cycle, vertex, graph):
    vertex_neighbor = graph.neighbors(vertex)
    neighbor_list = []
    for i in vertex_neighbor:
        if i in cycle:
            neighbor_list.append(i)
    return neighbor_list


def neighbour_in_cycle_vertex_2outer(cycle, vertex, graph):
    vertex_neighbor = graph.neighbors(vertex)

    for i in vertex_neighbor:
        if i in cycle and i != 'v_a' and i != 'v_b' and i != 'v_c':
            return i


def face_dual(graph):
    H1 = graph.to_directed()
    all_cycles1 = list(nx.simple_cycles(H1))
    all_triangles1 = []

    # print(all_cycles1)
    for cycle in all_cycles1:
        if len(cycle) > 2:
            all_triangles1.append(cycle)

    for i in all_triangles1:
        i.sort()

    unique_list = []

    for x in all_triangles1:
        if x not in unique_list:
            unique_list.append(x)

    print(unique_list)


# def recur_dfs(vertex1,cycle1,visited):
#     visited.add(vertex1)
#     new_cycle.append(vertex1)

def order_traverse(cycle_instance, graph_i):
    new_cycle_1 = []
    starting_vertex = cycle_instance[0]
    current_vertex = starting_vertex
    visited = set()
    new_cycle_1.append(starting_vertex)
    visited.add(starting_vertex)
    # print("---")
    # print(current_vertex)
    # print()
    # recur_dfs(starting_vertex,cycle_instance,visited)
    while len(visited) != len(cycle_instance):
        temp = neighbour_in_cycle_vertex_2(cycle_instance, current_vertex, graph_i)[0]
        temp2 = neighbour_in_cycle_vertex_2(cycle_instance, current_vertex, graph_i)[1]
        if temp not in visited:
            current_vertex = temp
            visited.add(current_vertex)
            new_cycle_1.append(current_vertex)
        else:
            current_vertex = temp2
            visited.add(current_vertex)
            new_cycle_1.append(current_vertex)

    return (new_cycle_1)


def order_traverse_outer(cycle_instance, graph_i):
    new_cycle_1 = []
    if 'v_a' in cycle_instance:
        starting_vertex = 'v_a'
    elif 'v_b' in cycle_instance:
        starting_vertex = 'v_b'
    else:
        starting_vertex = 'v_c'
    current_vertex = starting_vertex
    visited = set()
    new_cycle_1.append(starting_vertex)
    visited.add(starting_vertex)
    # print("---")
    # print(current_vertex)
    # print()
    # recur_dfs(starting_vertex,cycle_instance,visited)
    while len(visited) != len(cycle_instance):
        temp = neighbour_in_cycle_vertex_2(cycle_instance, current_vertex, graph_i)[0]
        temp2 = neighbour_in_cycle_vertex_2(cycle_instance, current_vertex, graph_i)[1]
        if temp not in visited:
            current_vertex = temp
            visited.add(current_vertex)
            new_cycle_1.append(current_vertex)
        else:
            current_vertex = temp2
            visited.add(current_vertex)
            new_cycle_1.append(current_vertex)

    return (new_cycle_1)


def get_outer_vertex(cycle_list):
    out = []
    if 'v_a' in cycle_list:
        out.append('v_a')
    if 'v_b' in cycle_list:
        out.append('v_b')
    if 'v_c' in cycle_list:
        out.append('v_c')
    return out


# def place_vertex2(grid_pos,cycle_instance,grid_visited,graph_instance):
#     for i in cycle_instance:
#         if i in grid_visited:
#             cycle_instance.remove(i)
#     for i in cycle_instance:
#         if
def check_subset(test_list, sub_list):
    flag = 0
    if (all(x in test_list for x in sub_list)):
        flag = 1
    return flag


def ret_pos(mylist, val_to_find, rows, cols):
    for i in range(rows):
        for j in range(cols):
            if mylist[i][j] == val_to_find:
                postn = [i, j]
    return (postn)


def add_col(mylist, rows):
    rows = len(mylist)
    cols = len(mylist[0])
    for i in range(rows):
        mylist[i].append(0)
    return mylist


def add_row(mylist, cols):
    mylist.append([0] * (cols + 1))
    return mylist


def shift_row(mylist, j, cols):
    mylist = add_row(mylist, cols)
    newlist = mylist[:j] + [[0] * cols] + mylist[j:-1]
    return newlist


def shift_col(mylist, j, rows):
    mylist = add_col(mylist, rows)
    newlist = [row[:j] + [0] + row[j:-1] for row in mylist]
    return newlist


def common_vertex(edge1, edge2):
    p1 = edge1[0]
    p2 = edge1[1]
    p3 = edge2[0]
    p4 = edge2[1]
    if p1 == p3 or p1 == p4:
        return p1
    elif p2 == p4 or p2 == p3:
        return p2


def point_between_hori(p1, p2, p3, new_pos):
    if new_pos[p1][0] < new_pos[p2][0] and new_pos[p2][0] < new_pos[p3][0]:
        new_pos[p3] = tuple((new_pos[p3][0], new_pos[p3][1] - 0.05))
    elif new_pos[p1][0] < new_pos[p3][0] and new_pos[p3][0] < new_pos[p2][0]:
        new_pos[p2] = tuple((new_pos[p2][0], new_pos[p2][1] - 0.05))
    elif new_pos[p1][0] > new_pos[p2][0] and new_pos[p2][0] > new_pos[p3][0]:
        new_pos[p1] = tuple((new_pos[p1][0], new_pos[p1][1] - 0.05))
    elif new_pos[p1][0] > new_pos[p3][0] and new_pos[p3][0] > new_pos[p2][0]:
        # new_pos[p1][1]=new_pos[p1][1]-0.05
        new_pos[p1] = tuple((new_pos[p1][0], new_pos[p1][1] - 0.05))

    return


def sort_edge_horiz(edge1):
    newList = []
    if (edge1[0][0] < edge1[1][0]):
        newList.append(edge1[0])
        newList.append(edge1[1])
    else:
        newList.append(edge1[1])
        newList.append(edge1[0])
    return newList


def check_intersection_utility_horizontal(edge1, edge2, new_pos):
    p1 = edge1[0]
    p2 = edge1[1]
    p3 = edge2[0]
    p4 = edge2[1]
    if new_pos[p1][0] == new_pos[p3][0]:
        point_between_hori(p1, p2, p4, new_pos)
    elif new_pos[p1][0] == new_pos[p4][0]:
        point_between_hori(p1, p2, p3, new_pos)
    elif new_pos[p2][0] == new_pos[p3][0]:
        point_between_hori(p2, p1, p4, new_pos)
    elif new_pos[p2][0] == new_pos[p4][0]:
        point_between_hori(p2, p1, p3, new_pos)
    else:
        edge1 = sort_edge_horiz(edge1)
        edge2 = sort_edge_horiz(edge2)
        p1 = edge1[0]
        p2 = edge1[1]
        p3 = edge2[0]
        p4 = edge2[1]
        if (new_pos[p1][0] > new_pos[p3][0] and new_pos[p2][0] < new_pos[p4][0]):  # edge1 inside edge2
            new_pos[p4] = tuple((new_pos[p4][0], new_pos[p4][1] - 0.05))
        elif (new_pos[p1][0] < new_pos[p3][0] and new_pos[p4][0] < new_pos[p2][0]):  # edge1 outside
            new_pos[p2] = tuple((new_pos[p2][0], new_pos[p2][1] - 0.05))


def check_intersection(edge_list, new_pos):
    for i in range(len(edge_list)):
        for j in range(i + 1, len(edge_list)):
            edge1 = edge_list[i]
            edge2 = edge_list[j]
            check_intersection_utility_horizontal(edge1, edge2, new_pos)

def common_in_list(list_instance):
   return list(reduce(set.intersection, [set(item) for item in list_instance]))

def rotate_grid(grid_graph,angle,grid_pos):
    rotated_pos={}
    if angle==90:
        for i in grid_pos:
            rotated_pos[i]=tuple((grid_pos[i][1],-1*grid_pos[i][0]))
        return rotated_pos

    elif angle==180:
        for i in grid_pos:
            rotated_pos[i]=tuple((grid_pos[i][0],-1*grid_pos[i][1]))
        return rotated_pos
    elif angle==270:
        for i in grid_pos:
            rotated_pos[i] = tuple((-1*grid_pos[i][1], grid_pos[i][0]))
        return rotated_pos
    else:
        return grid_pos
    return grid_pos

def merge_graph(graph1,graph2,pos1,pos2,dir1,dir2,point_eg,gm1,gm2):
    rotated_pos1={}
    rotated_pos2={}
    rows1=len(gm1)
    cols1=len(gm1[0])
    rows2=len(gm2)
    cols2=len(gm2[0])
    if dir1=='L':
        rotated_pos1=rotate_grid(graph1,270,pos1)
        print(gm1)
        temp = []
        for i in range(0, rows1):
            if gm1[i][0] != 0:
                temp.append(gm1[i][0])
        print(temp)
        print(graph1.edges)
        # for i in range(1,len(temp)):
        #     graph1.remove_edge(temp[i-1], temp)
        #     # del rotated_pos1[temp[i]]
        for i in graph1.edges:
            if i[0] in temp and i[1] in temp:
                graph1.remove_edge(i[0], i[1])
    elif dir1=='R':
        rotated_pos1=rotate_grid(graph1,90,pos1)
        print(gm1)
        temp = []
        for i in range(0, rows1):
            if gm1[i][cols1-1] != 0:
                temp.append(gm1[i][cols1-1])
        print(temp)
        print(graph1.edges)
        # for i in range(1,len(temp)):
        #     graph1.remove_edge(temp[i-1], temp)
        #     # del rotated_pos1[temp[i]]
        for i in graph1.edges:
            if i[0] in temp and i[1] in temp:
                graph1.remove_edge(i[0], i[1])
    elif dir1=='B':
        rotated_pos1=rotate_grid(graph1,0,pos1)
        print(gm1)
        temp=[]
        for i in range(0,cols1):
            if gm1[rows1-1][i]!=0:
               temp.append(gm1[rows1-1][i])
        print(temp)
        print(graph1.edges)
        # for i in range(1,len(temp)):
        #     graph1.remove_edge(temp[i-1], temp)
        #     # del rotated_pos1[temp[i]]
        for i in graph1.edges:
            if i[0] in temp and i[1] in temp:
                graph1.remove_edge(i[0],i[1])
    if dir2=='L':
        rotated_pos2=rotate_grid(graph2,90,pos2)
        print(gm2)
        temp = []
        for i in range(0, rows2):
            if gm2[i][0] != 0:
                temp.append(gm2[i][0])
        print(temp)
        print(graph2.edges)
        # for i in range(1,len(temp)):
        #     graph1.remove_edge(temp[i-1], temp)
        #     # del rotated_pos1[temp[i]]
        for i in graph2.edges:
            if i[0] in temp and i[1] in temp:
                graph2.remove_edge(i[0], i[1])
    elif dir2=='R':
        rotated_pos2=rotate_grid(graph2,270,pos2)
        print(gm2)
        temp = []
        for i in range(0, rows2):
            if gm2[i][cols-1] != 0:
                temp.append(gm2[i][cols2-1])
        print(temp)
        print(graph2.edges)
        # for i in range(1,len(temp)):
        #     graph1.remove_edge(temp[i-1], temp)
        #     # del rotated_pos1[temp[i]]
        for i in graph2.edges:
            if i[0] in temp and i[1] in temp:
                graph2.remove_edge(i[0], i[1])
    elif dir2=='B':
        rotated_pos2=rotate_grid(graph2,180,pos2)
        print(gm2)
        temp = []
        for i in range(0, cols2):
            if gm2[rows2 - 1][i] != 0:
                temp.append(gm2[rows2 - 1][i])
        print(temp)
        print(graph2.edges)
        # for i in range(1,len(temp)):
        #     graph1.remove_edge(temp[i-1], temp)
        #     # del rotated_pos1[temp[i]]
        for i in graph2.edges:
            if i[0] in temp and i[1] in temp:
                graph2.remove_edge(i[0], i[1])

    print(rotated_pos1)
    print(rotated_pos2)
    print(rotated_pos1[point_eg])
    diff_x=rotated_pos1[point_eg][0]-rotated_pos2[point_eg][0]
    diff_y=rotated_pos1[point_eg][1]-rotated_pos2[point_eg][1]
    nx.draw(graph1, rotated_pos1, with_labels=False)
    plt.show()
    print(diff_x)
    print(diff_y)
    nx.draw(graph2, rotated_pos2, with_labels=False)
    plt.show()
    translated_pos1={}
    translated_pos2={}
    for i in rotated_pos2:
        translated_pos2[i]=tuple((rotated_pos2[i][0]+diff_x,rotated_pos2[i][1]+diff_y))
    nx.draw(graph1, rotated_pos1, with_labels=False)
    nx.draw(graph2, translated_pos2, with_labels=False)
    plt.show()

def grid_drawing2(counter):
    new_cycle = []
    G = nx.Graph()
    number_of_vertices = int(input("Enter number of vertices for graph #" + str(counter) + " : "))
    while number_of_vertices > 0:
        number_of_vertices = number_of_vertices - 1
        strr = input("Enter: ")
        tmp = strr.split(":")

        for x in tmp[1].split(','):
            G.add_edge(tmp[0], x)
    origin_pos = nx.planar_layout(G)

    for nodeid in origin_pos:
        G.add_node(nodeid, pos=(origin_pos[nodeid][0], origin_pos[nodeid][1]))

    origin_pos = nx.get_node_attributes(G, 'pos')

    nx.draw(G, origin_pos, with_labels=True)
    plt.show()

    myset = face_list(G,origin_pos)
    outerfacesfinal = outer_faces(G, myset, G.number_of_nodes(),origin_pos)
    face_edge_outer = outer_faces_edge_mapping(G, myset, G.number_of_nodes(),origin_pos)
    mylist = list(myset)
    vertex_to_face_map = {}
    pos0 = []
    geodual = nx.Graph()
    for i in myset:
        posx = tuple(map(sum, zip(origin_pos[i[0]], origin_pos[i[1]], origin_pos[i[2]])))
        posx = tuple(posxe / 3 for posxe in posx)
        # print(i)
        # print(posx)
        if i in outerfacesfinal:
            pos0.append(posx)
        # print(mylist)
        vertex_to_face_map['v' + str(mylist.index(i))] = i
        geodual.add_node(mylist.index(i), pos=posx)
    # print(vertex_to_face_map)
    for i in mylist:
        for j in mylist:
            if i < j:
                if common_edge(i, j):
                    geodual.add_edge(mylist.index(i), mylist.index(j))

    pos = nx.get_node_attributes(geodual, 'pos')

    pos_0 = np.array(pos0[0])
    pos_1 = np.array(pos0[1])
    pos_2 = np.array(pos0[2])

    numbernodes = geodual.number_of_nodes()
    geodual.add_node(numbernodes)
    geodual.add_node(numbernodes + 1)
    geodual.add_node(numbernodes + 2)

    vertex_to_face_map['v_a'] = list(face_edge_outer[tuple(outerfacesfinal[0])])
    vertex_to_face_map['v_b'] = list(face_edge_outer[tuple(outerfacesfinal[2])])
    vertex_to_face_map['v_c'] = list(face_edge_outer[tuple(outerfacesfinal[1])])

    # print(vertex_to_face_map)
    pos[numbernodes] = pos_0 + ((pos_0 - ((pos_2 + pos_1) / 2)) / 1)
    pos[numbernodes + 2] = pos_1 + ((pos_1 - ((pos_2 + pos_0) / 2)) / 1)
    pos[numbernodes + 1] = pos_2 + ((pos_2 - ((pos_0 + pos_1) / 2)) / 1)

    geodual.add_edge(numbernodes, mylist.index(outerfacesfinal[0]))
    geodual.add_edge(numbernodes + 1, mylist.index(outerfacesfinal[2]))
    geodual.add_edge(numbernodes + 2, mylist.index(outerfacesfinal[1]))
    geodual.add_edge(numbernodes, numbernodes + 1)
    geodual.add_edge(numbernodes + 0, numbernodes + 2)
    geodual.add_edge(numbernodes + 1, numbernodes + 2)

    labeldict = {}
    for i in range(geodual.number_of_nodes()):
        if i == numbernodes:
            labeldict[i] = "v_a"
        elif i == numbernodes + 1:
            labeldict[i] = "v_b"
        elif i == numbernodes + 2:
            labeldict[i] = "v_c"
        else:
            labeldict[i] = "v" + str(i)

    geodual = nx.relabel_nodes(geodual, labeldict, copy=False)
    temp = list(pos.keys())

    for i in temp:
        if i == numbernodes:
            pos["v_a"] = pos[i]
        elif i == numbernodes + 1:
            pos["v_b"] = pos[i]
        elif i == numbernodes + 2:
            pos["v_c"] = pos[i]
        else:
            pos["v" + str(i)] = pos[i]
        del (pos[i])

    # nx.draw(geodual, pos, with_labels=True)
    # plt.show()

    H1 = geodual.to_directed()
    all_cycles1 = list(nx.simple_cycles(H1))
    all_triangles1 = []

    # print(all_cycles1)
    for cycle in all_cycles1:
        if len(cycle) > 2:
            all_triangles1.append(cycle)

    for i in all_triangles1:
        i.sort()

    unique_list = []

    for x in all_triangles1:
        if x not in unique_list:
            unique_list.append(x)

    # print(unique_list)
    unique_list.sort(key=len, reverse=True)
    # print(unique_list)

    chordless_cycle_list = []
    for i in unique_list:
        if not chordless_cycle(i, geodual):
            chordless_cycle_list.append(i)
    # print(chordless_cycle_list)

    #
    chordless_cycle_list_remove = set()
    for cycle_instance in chordless_cycle_list:

        flag_inside = 0
        coords = []
        order_cycle_instance = order_traverse(cycle_instance, geodual)
        for j in order_cycle_instance:
            coords.append(pos[j])
        poly = Polygon(coords)
        for i in geodual.nodes:
            if not i in cycle_instance:
                p1 = Point(pos[i][0], pos[i][1])
                if p1.within(poly):
                    # print("inside")
                    # print(p1)

                    flag_inside = 1
                    break
        if flag_inside:
            chordless_cycle_list_remove.add(chordless_cycle_list.index(cycle_instance))

    for ele in sorted(chordless_cycle_list_remove, reverse=True):
        del chordless_cycle_list[ele]

    ordered_chordless_cycle = []
    for i in chordless_cycle_list:
        order_cycle_instance2 = order_traverse(i, geodual)
        ordered_chordless_cycle.append(order_cycle_instance2)
    # print("len of chordless cycle")
    # print(len(ordered_chordless_cycle))
    print(ordered_chordless_cycle)

    # print(final_mapping)
    ordered_chordless_cycle.sort(key=len, reverse=True)

    result_outer_cycle = []

    for i in ordered_chordless_cycle:
        if 'v_a' in i or 'v_b' in i or 'v_c' in i:
            # print("max length")
            result_outer_cycle.append(i)
            # print(i)

    outer_final_mapping = {}
    print(result_outer_cycle)

    for inst in result_outer_cycle:
        temp_list = []
        for i in inst:
            temp_list.append(vertex_to_face_map[i])
        outer_final_mapping[tuple(inst)] = common_in_list(temp_list)

    print("outer_final_mapping")
    print(outer_final_mapping)
    outer_final_mapping2 = {}
    for i in outer_final_mapping:
        if 'v_a' in i and 'v_b' in i:
            temp = []
            temp.append('v_a')
            temp.append('v_b')
            outer_final_mapping2[outer_final_mapping[i][0]] = temp
        elif 'v_a' in i and 'v_c' in i:
            temp = []
            temp.append('v_a')
            temp.append('v_c')
            outer_final_mapping2[outer_final_mapping[i][0]] = temp
        elif 'v_c' in i and 'v_b' in i:
            temp = []
            temp.append('v_b')
            temp.append('v_c')
            outer_final_mapping2[outer_final_mapping[i][0]] = temp
    final_mapping = {}
    for inst in ordered_chordless_cycle:
        temp_list = []
        for i in inst:
            temp_list.append(vertex_to_face_map[i])
            final_mapping[tuple(inst)] = common_in_list(temp_list)
    print(final_mapping)
    # for i in final_mapping:
    #     counter=len(i)
    #     pos_sum=0
    #     for ine in i:
    #         pos_sum=pos_sum+pos[ine]
    #     pos_sum=pos_sum/counter

    nx.draw(geodual, pos, with_labels=True)
    matplotlib.pyplot.text(0.5, 0.5, 'matplotlib')
    plt.show()
        # text()
    # result_outer_cycle=[['v1', 'v6', 'v7', 'v3', 'v_b', 'v_c'],['v0', 'v4', 'v5', 'v3', 'v_b', 'v_a'],  ['v0', 'v2', 'v1', 'v_c', 'v_a']]
    # print(origin_pos)
    grid_pos = {}
    first_outer_cycle = result_outer_cycle[0]
    second_outer_cycle = result_outer_cycle[1]
    third_outer_cycle = result_outer_cycle[2]
    x_coord = 0.0
    y_coord = 0.0
    first_outer_cycle2 = first_outer_cycle.copy()
    second_outer_cycle2 = second_outer_cycle.copy()
    third_outer_cycle2 = third_outer_cycle.copy()
    grid_draw = nx.Graph()
    orexample = order_traverse_outer(first_outer_cycle, geodual)
    grid_visited = []
    # print(orexample)

    cols = len(first_outer_cycle)
    rows = len(second_outer_cycle2) - 1
    grid_matrix = [[0 for i in range(cols)] for j in range(rows)]
    temprow = 0
    tempcol = 0
    curved_outer_cycle = []
    curved_outer_cycle.append(first_outer_cycle2)
    for i in orexample:
        grid_pos[i] = tuple((x_coord, y_coord))
        x_coord = x_coord + 0.1
        grid_draw.add_node(i, pos=grid_pos)
        grid_visited.append(i)
        grid_matrix[temprow][tempcol] = i
        tempcol = tempcol + 1

    if orexample[0] in second_outer_cycle:
        curved_outer_cycle.append(second_outer_cycle2)
        curved_outer_cycle.append(third_outer_cycle2)
        while len(second_outer_cycle) != 0:
            for i in second_outer_cycle:
                if i in grid_visited:
                    second_outer_cycle.remove(i)
                else:
                    for ig in grid_visited:
                        if ig != 'v_a' and ig != 'v_b' and ig != 'v_c':
                            if ig in neighbour_in_cycle_vertex_2(second_outer_cycle2, i, geodual):
                                # xa=grid_pos[ig][0]
                                # ya=grid_pos[ig][1]

                                # grid_pos[i] = tuple((xa, ya-0.1))
                                grid_visited.append(i)
                                # grid_draw.add_node(i, pos=grid_pos)
                                point_pos = ret_pos(grid_matrix, ig,rows,cols)
                                grid_matrix[point_pos[0] + 1][point_pos[1]] = i
                                break

        while len(third_outer_cycle) != 0:
            for i in third_outer_cycle:
                if i in grid_visited:
                    third_outer_cycle.remove(i)
                else:
                    for ig in grid_visited:
                        if ig != 'v_a' and ig != 'v_b' and ig != 'v_c':
                            if ig in second_outer_cycle2:
                                continue
                            if ig in neighbour_in_cycle_vertex_2(third_outer_cycle2, i, geodual):
                                # xa=grid_pos[ig][0]
                                # ya=grid_pos[ig][1]

                                # grid_pos[i] = tuple((xa, ya-0.1))
                                grid_visited.append(i)
                                # grid_draw.add_node(i, pos=grid_pos)
                                point_pos = ret_pos(grid_matrix, ig,rows,cols)
                                grid_matrix[point_pos[0] + 1][point_pos[1]] = i
                                break

    else:
        curved_outer_cycle.append(third_outer_cycle2)
        curved_outer_cycle.append(second_outer_cycle2)
        while len(third_outer_cycle) != 0:
            for i in third_outer_cycle:
                if i in grid_visited:
                    third_outer_cycle.remove(i)
                else:
                    for ig in grid_visited:
                        if ig != 'v_a' and ig != 'v_b' and ig != 'v_c':
                            if ig in neighbour_in_cycle_vertex_2(third_outer_cycle2, i, geodual):
                                # xa=grid_pos[ig][0]
                                # ya=grid_pos[ig][1]

                                # grid_pos[i] = tuple((xa, ya-0.1))
                                grid_visited.append(i)
                                # grid_draw.add_node(i, pos=grid_pos)
                                point_pos = ret_pos(grid_matrix, ig,rows,cols)
                                grid_matrix[point_pos[0] + 1][point_pos[1]] = i
                                break
        while len(second_outer_cycle) != 0:
            for i in second_outer_cycle:
                if i in grid_visited:
                    second_outer_cycle.remove(i)
                else:
                    for ig in grid_visited:
                        if ig != 'v_a' and ig != 'v_b' and ig != 'v_c':
                            if ig in third_outer_cycle2:
                                continue
                            if ig in neighbour_in_cycle_vertex_2(second_outer_cycle2, i, geodual):
                                # xa=grid_pos[ig][0]
                                # ya=grid_pos[ig][1]
                                point_pos = ret_pos(grid_matrix, ig,rows,cols)
                                grid_matrix[point_pos[0] + 1][point_pos[1]] = i
                                # grid_pos[i] = tuple((xa, ya-0.1))
                                grid_visited.append(i)
                                # grid_draw.add_node(i, pos=grid_pos)
                                break

    temprow = 0
    tempcol = 1
    current_dir = 0
    horflag = 0
    vertflag = 0
    while len(grid_visited) != len(geodual.nodes):

        # 0 for horizontal 1 for vertical
        if current_dir == 0:
            for i in range(cols):
                if grid_matrix[temprow][i] == 0:
                    continue
                for k in geodual.neighbors(grid_matrix[temprow][i]):
                    if k not in grid_visited:
                        horflag = 1
                        if grid_matrix[temprow + 1][i] == 0:
                            grid_matrix[temprow + 1][i] = k
                            # temp_ele=grid_matrix[temprow][i]
                            # xa = grid_pos[temp_ele][0]
                            # ya = grid_pos[temp_ele][1]
                            grid_visited.append(k)
                            # grid_pos[k] = tuple((xa, ya - 0.1))
                            # grid_draw.add_node(k,pos=grid_pos)
                            break
                        else:
                            grid_matrix = shift_row(grid_matrix, temprow + 1)
                            rows = rows + 1
                            grid_matrix[temprow + 1][i] = k
                            grid_visited.append(k)
                            break
                            # for irow in range(temprow+2,rows):
                            #     if grid_matrix[irow][i]==0:
                            #         continue
                            # grid_pos[grid_matrix[irow][i]]=tuple((grid_pos[grid_matrix[irow][i]][0],grid_pos[grid_matrix[irow][i]][1]-0.1))

            if horflag == 0:
                current_dir = 0
            else:
                horflag = 0
                current_dir = 1
            temprow = temprow + 1
        else:
            for i in range(rows):
                if grid_matrix[i][tempcol] == 0:
                    continue
                for k in geodual.neighbors(grid_matrix[i][tempcol]):
                    if k not in grid_visited:
                        vertflag = 1
                        if grid_matrix[i][tempcol + 1] == 0:
                            grid_matrix[i][tempcol + 1] = k
                            # temp_ele=grid_matrix[i][tempcol]
                            # xa = grid_pos[temp_ele][0]
                            # ya = grid_pos[temp_ele][1]
                            grid_visited.append(k)
                            # grid_pos[k] = tuple((xa+0.1, ya ))
                            # grid_draw.add_node(k,pos=grid_pos)
                            break
                        else:
                            grid_matrix = shift_col(grid_matrix, tempcol + 1)
                            cols = cols + 1
                            grid_matrix[i][tempcol + 1] = k
                            grid_visited.append(k)
                            break
                            # temp_ele = grid_matrix[i][tempcol]
                            # xa = grid_pos[temp_ele][0]
                            # ya = grid_pos[temp_ele][1]
                            # grid_pos[k] = tuple((xa + 0.1, ya))
                            # grid_draw.add_node(k, pos=grid_pos)

                            # for icol in range(tempcol + 2, cols):
                            #     if grid_matrix[i][icol]==0:
                            #         continue
                            # grid_pos[grid_matrix[i][icol]] = tuple(
                            #     (grid_pos[grid_matrix[i][icol]][0]+0.1, grid_pos[grid_matrix[i][icol]][1]))
            if vertflag == 0:
                current_dir = 1
            else:
                vertflag = 0
                current_dir = 0
            tempcol = tempcol + 1
    # print(grid_matrix)
    new_pos = {}
    test_graph = nx.Graph()
    xcordnew = 0.0
    ycordnew = 0.0
    for i in range(rows):
        for j in range(cols):
            if grid_matrix[i][j] != 0:
                new_pos[grid_matrix[i][j]] = tuple((xcordnew, ycordnew))
                test_graph.add_node(grid_matrix[i][j], pos=new_pos)
            xcordnew = xcordnew + 0.1
        ycordnew = ycordnew - 0.1
        xcordnew = 0.0

    # print(grid_pos)
    # print("edges")
    # print(geodual.edges())

    nx.draw(test_graph, new_pos, with_labels=True)
    # print(new_pos)
    plt.show()

    direction_mapping={}
    first_edge_outer_vertex = get_outer_vertex(curved_outer_cycle[0])
    direction_mapping[tuple(first_edge_outer_vertex)]='U'
    second_edge_outer_vertex = get_outer_vertex(curved_outer_cycle[1])
    third_edge_outer_vertex = get_outer_vertex(curved_outer_cycle[2])
    direction_mapping[tuple(second_edge_outer_vertex)]='L'
    direction_mapping[tuple(second_edge_outer_vertex)]='L'
    direction_mapping[tuple(third_edge_outer_vertex)]='R'
    x_a = first_edge_outer_vertex[0]
    x_b = first_edge_outer_vertex[1]
    grid_matrix=shift_row(grid_matrix,0,cols)
    rows=rows+1
    grid_matrix[0][0]='n1'
    grid_matrix[0][cols-1]='n2'
    new_pos['n' + str(1)] = tuple((new_pos[x_a][0], new_pos[x_a][1] + 0.1))
    new_pos['n' + str(2)] = tuple((new_pos[x_b][0], new_pos[x_b][1] + 0.1))
    test_graph.add_node('n1', pos=new_pos)
    test_graph.add_node('n2', pos=new_pos)

    temp_list = (curved_outer_cycle[0]).copy()
    temp_list.append('n1')
    temp_list.append('n2')
    final_mapping[tuple(temp_list)] = final_mapping[tuple(curved_outer_cycle[0])]
    del final_mapping[tuple(curved_outer_cycle[0])]
    # print(final_mapping)
    test_graph.add_edge('n1', x_a)
    test_graph.add_edge('n2', x_b)
    test_graph.add_edge('n1', 'n2')
    second_edge_outer_vertex = get_outer_vertex(curved_outer_cycle[1])
    x_1 = second_edge_outer_vertex[0]
    # y_1=second_edge_outer_vertex[0][1]
    x_2 = second_edge_outer_vertex[1]
    # y_2=second_edge_outer_vertex[1][1]
    if new_pos[x_1][1] < new_pos[x_2][1]:
        new_pos['n' + str(3)] = tuple((new_pos[x_2][0], new_pos[x_1][1]))
    else:
        new_pos['n' + str(3)] = tuple((new_pos[x_1][0], new_pos[x_2][1]))
    test_graph.add_node('n3', pos=new_pos)
    grid_matrix[rows-1][0]='n3'
    test_graph.add_edge('n3', x_1)
    test_graph.add_edge('n3', x_2)

    temp_list = (curved_outer_cycle[1]).copy()
    temp_list.append('n3')
    final_mapping[tuple(temp_list)] = final_mapping[tuple(curved_outer_cycle[1])]
    del final_mapping[tuple(curved_outer_cycle[1])]

    third_edge_outer_vertex = get_outer_vertex(curved_outer_cycle[2])
    x_1 = third_edge_outer_vertex[0]
    # y_1=second_edge_outer_vertex[0][1]
    x_2 = third_edge_outer_vertex[1]
    # y_2=second_edge_outer_vertex[1][1]
    if new_pos[x_1][1] < new_pos[x_2][1]:
        new_pos['n' + str(4)] = tuple((new_pos[x_2][0], new_pos[x_1][1]))
    else:
        new_pos['n' + str(4)] = tuple((new_pos[x_1][0], new_pos[x_2][1]))
    test_graph.add_node('n4', pos=new_pos)
    grid_matrix[rows-1][cols-1]='n4'
    test_graph.add_edge('n4', x_1)
    test_graph.add_edge('n4', x_2)

    temp_list = (curved_outer_cycle[2]).copy()
    temp_list.append('n4')
    final_mapping[tuple(temp_list)] = final_mapping[tuple(curved_outer_cycle[2])]
    del final_mapping[tuple(curved_outer_cycle[2])]
    # print(final_mapping)
    print(grid_matrix)

    counter = 5
    final_row = len(grid_matrix)
    final_col = len(grid_matrix[0])
    # print(final_col)
    # print(final_row)
    # print(grid_matrix)
    # print("edge list")
    # print(geodual.edges)
    for i in range(1, final_row):
        rowelements = []
        for j in range(1, final_col - 1):
            rowelements.append(grid_matrix[i][j])
        edges_here = []
        for ie in geodual.edges:
            if ie[0] in rowelements and ie[1] in rowelements:
                edges_here.append(ie)
        # print("rowelements_here")
        # print(edges_here)
        # print(rowelements)
        if edges_here != []:
            # print("hi")
            check_intersection(edges_here,new_pos)    #this shifts vertices so that edges dont intersect

    for i in geodual.edges():
        if i[0] == 'v_a' and i[1] == 'v_b' or i[0] == 'v_b' and i[1] == 'v_c' or i[0] == 'v_a' and i[1] == 'v_c':
            continue
        x1 = new_pos[i[0]][0]
        y1 = new_pos[i[0]][1]
        x2 = new_pos[i[1]][0]
        y2 = new_pos[i[1]][1]
        p1 = i[0]
        p2 = i[1]
        if (x1 == x2 or y1 == y2):
            test_graph.add_edge(i[0], i[1])
        else:
            # print(p1)
            # print(p2)
            new_vertex = 'n' + str(counter)
            if new_pos[p1][1] < new_pos[p2][1]:
                point_exist_flag = 0
                point_exist_flag2 = 0
                for i in new_pos:
                    if new_pos[i][0] == new_pos[p2][0] and new_pos[i][1] == new_pos[p1][1]:
                        point_exist_flag = 1
                    if new_pos[i][0] == new_pos[p1][0] and new_pos[i][1] == new_pos[p2][1]:
                        point_exist_flag2 = 1
                if point_exist_flag == 0:
                    new_pos['n' + str(counter)] = tuple((new_pos[p2][0], new_pos[p1][1]))
                elif point_exist_flag2 == 0:
                    new_pos['n' + str(counter)] = tuple((new_pos[p1][0], new_pos[p2][1]))
                else:
                    new_pos[p2] = tuple((new_pos[p2][0], new_pos[p2][1] - 0.05))
                    new_pos['n' + str(counter)] = tuple((new_pos[p1][0], new_pos[p2][1]))

            else:
                point_exist_flag = 0
                point_exist_flag2 = 0
                for i in geodual.nodes:
                    if new_pos[i][0] == new_pos[p1][0] and new_pos[i][1] == new_pos[p2][1]:
                        point_exist_flag = 1
                    if new_pos[i][0] == new_pos[p2][0] and new_pos[i][1] == new_pos[p1][1]:
                        point_exist_flag2 = 1
                if point_exist_flag == 0:
                    new_pos['n' + str(counter)] = tuple((new_pos[p1][0], new_pos[p2][1]))
                elif point_exist_flag2 == 0:
                    new_pos['n' + str(counter)] = tuple((new_pos[p2][0], new_pos[p1][1]))
                else:
                    new_pos[p1] = tuple((new_pos[p1][0], new_pos[p1][1] - 0.05))
                    new_pos['n' + str(counter)] = tuple((new_pos[p2][0], new_pos[p1][1]))
            test_graph.add_node(new_vertex, pos=new_pos)
            test_graph.add_edge(new_vertex, p1)
            test_graph.add_edge(new_vertex, p2)
            counter = counter + 1

    # print(new_pos)
    # nx.draw(test_graph, new_pos, with_labels=True)
    # plt.show()
    return test_graph,new_pos,outer_final_mapping2,grid_matrix,direction_mapping


number_of_graphs = int(input("Enter number of graphs: "))
counter = 1
graph_list = []
pos_list = []
map_list=[]
grid_matrix_list=[]
dir_list=[]
while (number_of_graphs) > 0:
    number_of_graphs = number_of_graphs - 1
    grid_graph, graph_pos,final_map,grid_matrix,direction_mapping = grid_drawing2(counter)
    grid_matrix_list.append(grid_matrix)
    graph_list.append(grid_graph)
    pos_list.append(graph_pos)
    map_list.append(final_map)
    dir_list.append(direction_mapping)
    nx.draw(grid_graph, graph_pos, with_labels=True)
    plt.show()

    # print(graph_pos)
    # print("Region-Vertex mapping")
    # print(final_map)
    plt.show()
    counter = counter + 1
common_vertex=[]
for i in map_list[0]:
    for j in map_list[1]:
        if i==j:
            common_vertex.append(i)

print(dir_list)
print(map_list)
print(common_vertex)
dir1=dir_list[0]
dir2=dir_list[1]
map1=map_list[0]
map2=map_list[1]
comv1=common_vertex[0]
comv2=common_vertex[1]
print(dir1[tuple(map1[comv1])])
print(dir1[tuple(map1[comv2])])
print(dir2[tuple(map2[comv1])])
print(dir2[tuple(map2[comv2])])

if dir1[tuple(map1[comv1])]=='L' and dir1[tuple(map1[comv2])]=='R':
    var1='B'
elif dir1[tuple(map1[comv2])]=='L' and dir1[tuple(map1[comv1])]=='R':
    var1='B'
elif dir1[tuple(map1[comv1])]=='U' and dir1[tuple(map1[comv2])]=='R':
    var1='R'
elif dir1[tuple(map1[comv2])]=='U' and dir1[tuple(map1[comv1])]=='R':
    var1='R'
elif dir1[tuple(map1[comv1])]=='U' and dir1[tuple(map1[comv2])]=='L':
    var1='L'
elif dir1[tuple(map1[comv2])]=='U' and dir1[tuple(map1[comv1])]=='L':
    var1='L'

if dir2[tuple(map2[comv1])]=='L' and dir2[tuple(map2[comv2])]=='R':
    var2='B'
elif dir2[tuple(map2[comv2])]=='L' and dir2[tuple(map2[comv1])]=='R':
    var2='B'
elif dir2[tuple(map2[comv1])]=='U' and dir2[tuple(map2[comv2])]=='R':
    var2='R'
elif dir2[tuple(map2[comv2])]=='U' and dir2[tuple(map2[comv1])]=='R':
    var2='R'
elif dir2[tuple(map2[comv1])]=='U' and dir2[tuple(map2[comv2])]=='L':
    var2='L'
elif dir2[tuple(map2[comv2])]=='U' and dir2[tuple(map2[comv1])]=='L':
    var2='L'
print(map_list)
print(dir_list)
grid=grid_matrix_list[0]
rows=len(grid)
cols=len(grid[0])
if var1=='L':
    var3=grid[0][0]
elif var1=='R':
    var3=grid[rows-1][cols-1]
elif var1=='B':
    var3=grid[rows-1][0]
print("VAR")
print(var1)
print(var2)
print(var3)
merge_graph(graph_list[0],graph_list[1],pos_list[0],pos_list[1],var1,var2,var3,grid_matrix_list[0],grid_matrix_list[1])


