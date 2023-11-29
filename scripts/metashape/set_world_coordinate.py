#  Copyright Jian Wang @ MPI-INF (c) 2023.

def set_world_coordinate():
    def cross(a, b):
        result = Metashape.Vector([a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x])
        return result.normalized()

    chunk = Metashape.app.document.chunk
    # suppose we have three points as the markers
    marker_list = chunk.markers
    for marker in marker_list:
        if marker.label == 'point 1':
            m1 = marker
        if marker.label == 'point 2':
            m2 = marker
        if marker.label == 'point 3':
            m3 = marker

    X = (m2.position - m1.position).normalized()
    Y = (m3.position - m1.position).normalized()
    Z = cross(X, Y)
    Y1 = -cross(X, Z)
    x_o = m1.position[0]
    y_o = m1.position[1]
    z_o = m1.position[2]

    T = Metashape.Matrix([[X.x, Y1.x, Z.x, x_o], [X.y, Y1.y, Z.y, y_o], [X.z, Y1.z, Z.z, z_o], [0, 0, 0, 1]]).inv()
    chunk.transform.matrix = T
