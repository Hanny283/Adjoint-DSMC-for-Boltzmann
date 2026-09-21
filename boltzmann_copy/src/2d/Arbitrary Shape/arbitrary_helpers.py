import numpy as np 
import pygmsh 
from edge_class import Edge
import cell_class as ct
from scipy.spatial import cKDTree


class CachedBoundary:
    """
    Cached boundary structure for efficient boundary condition processing.
    
    Precomputes and caches:
    - Boundary edges
    - Edge normals
    - KD-tree for closest edge lookup
    - Edge segment array for vectorized operations
    """
    def __init__(self, boundary_points):
        """
        Initialize cached boundary structure.
        
        Parameters
        ----------
        boundary_points : array-like of shape (N, 2)
            Boundary point coordinates
        """
        self.boundary_points = np.asarray(boundary_points)
        self._edges = None
        self._edge_segments = None
        self._edge_normals = None
        self._edge_midpoints = None
        self._kdtree = None
        self._compute_cache()
    
    def _compute_cache(self):
        """Compute and cache all boundary data structures."""
        # Get boundary edges
        self._edges = get_boundary_edges(self.boundary_points)
        
        # Convert to array format [x1, y1, x2, y2] for each edge
        num_edges = len(self._edges)
        self._edge_segments = np.zeros((num_edges, 4))
        self._edge_normals = np.zeros((num_edges, 2))
        self._edge_midpoints = np.zeros((num_edges, 2))
        
        for i, edge in enumerate(self._edges):
            x1, y1, x2, y2 = edge
            self._edge_segments[i] = [x1, y1, x2, y2]
            
            # Compute normal
            normal_x, normal_y = get_edge_normal(x1, y1, x2, y2)
            self._edge_normals[i] = [normal_x, normal_y]
            
            # Compute midpoint for KD-tree
            self._edge_midpoints[i] = [(x1 + x2) / 2, (y1 + y2) / 2]
        
        # Build KD-tree on edge midpoints for fast closest edge lookup
        if len(self._edge_midpoints) > 0:
            self._kdtree = cKDTree(self._edge_midpoints)
    
    @property
    def edges(self):
        """Get boundary edges."""
        return self._edges
    
    @property
    def edge_segments(self):
        """Get edge segments as array [x1, y1, x2, y2]."""
        return self._edge_segments
    
    @property
    def edge_normals(self):
        """Get edge normals as array."""
        return self._edge_normals
    
    def find_closest_edges(self, positions, k=1):
        """
        Find closest edge(s) for each position using KD-tree.
        
        Parameters
        ----------
        positions : ndarray of shape (N, 2)
            Positions to find closest edges for
        k : int, optional
            Number of closest edges to return (default: 1)
            
        Returns
        -------
        edge_indices : ndarray of shape (N, k)
            Indices of closest edges
        distances : ndarray of shape (N, k)
            Distances to edge midpoints (approximate)
        """
        if self._kdtree is None or len(positions) == 0:
            return np.array([]), np.array([])
        
        distances, indices = self._kdtree.query(positions, k=min(k, len(self._edge_segments)))
        
        if k == 1:
            indices = indices.reshape(-1, 1)
            distances = distances.reshape(-1, 1)
        
        return indices, distances
    
    def get_closest_edge_info(self, positions):
        """
        Get closest edge information (distance and closest point) for positions.
        
        Uses KD-tree to find candidate edges, then computes exact distances.
        
        Parameters
        ----------
        positions : ndarray of shape (N, 2)
            Positions to process
            
        Returns
        -------
        edge_indices : ndarray of shape (N,)
            Index of closest edge for each position
        distances : ndarray of shape (N,)
            Exact distance to closest edge
        closest_points : ndarray of shape (N, 2)
            Closest point on edge for each position
        normals : ndarray of shape (N, 2)
            Normal vector for closest edge
        """
        if len(positions) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Find candidate edges using KD-tree (check 3 closest for accuracy)
        candidate_indices, _ = self.find_closest_edges(positions, k=min(3, len(self._edge_segments)))
        
        N = len(positions)
        edge_indices = np.zeros(N, dtype=int)
        distances = np.full(N, np.inf)
        closest_points = np.zeros((N, 2))
        normals = np.zeros((N, 2))
        
        # For each position, check candidate edges and find true closest
        for i, pos in enumerate(positions):
            candidates = candidate_indices[i]
            min_dist = np.inf
            best_idx = 0
            best_point = None
            
            for edge_idx in candidates:
                edge = self._edge_segments[edge_idx]
                x1, y1, x2, y2 = edge
                dist, closest_pt = point_to_line_distance(pos[0], pos[1], x1, y1, x2, y2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_idx = edge_idx
                    best_point = closest_pt
            
            edge_indices[i] = best_idx
            distances[i] = min_dist
            closest_points[i] = best_point
            normals[i] = self._edge_normals[best_idx]
        
        return edge_indices, distances, closest_points, normals

def create_arbitrary_shape_mesh_2d(N,points, mesh_size=0.1):
    """
    Create an arbitrary shape in 2D using B-spline curves for smooth boundaries.
    
    Parameters
    ----------
    N : int
        Number of particles (not used for mesh generation, kept for compatibility)
    points : array-like
        Boundary points defining the shape
    mesh_size : float, optional
        Characteristic mesh size. Larger values = fewer, larger cells.
        Default is 0.1. Typical range: 0.05 (fine) to 0.5 (coarse).
    
    Returns
    -------
    mesh : pygmsh mesh object
        Generated triangular mesh
    """

    with pygmsh.occ.Geometry() as geom:

        # 1) Add gmsh points with mesh size
        gmpts = [geom.add_point([x, y], mesh_size=mesh_size) for (x, y) in points]

        # 2) Create B-spline curve connecting all points
        # For a closed B-spline, we need to repeat the first point at the end
        closed_points = gmpts + [gmpts[0]]  # Close the curve
        
        # Create B-spline curve with the closed point list
        bspline_curve = geom.add_bspline(closed_points)

        # 3) Close the boundary: use a curve loop, then a plane surface
        loop = geom.add_curve_loop([bspline_curve])
        surf = geom.add_plane_surface(loop)

        # tag the boundary as a Physical Group for later lookup
        boundary_pg = geom.add_physical([bspline_curve], label="boundary")
        domain_pg   = geom.add_physical([surf], label="domain")

        # 4) Generate mesh (triangles by default)
        mesh = geom.generate_mesh()

    return mesh

def assign_positions_arbitrary_2d(N, mesh):

    # 1) Extract triangle data from mesh
    triangle_indices = None
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangle_indices = cell.data
            break
    if triangle_indices is None:
        raise ValueError("No triangles found.")

    mesh_points = mesh.points  # x,y coordinates of all mesh points

    # 2) Calculate triangle areas using cross product
    vertex_a = mesh_points[triangle_indices[:, 0]]  # First vertex of each triangle
    vertex_b = mesh_points[triangle_indices[:, 1]]  # Second vertex of each triangle
    vertex_c = mesh_points[triangle_indices[:, 2]]  # Third vertex of each triangle
    
    # Calculate areas using 2D cross product formula
    triangle_areas = 0.5 * np.abs((vertex_b[:, 0] - vertex_a[:, 0]) * (vertex_c[:, 1] - vertex_a[:, 1]) - 
                                  (vertex_c[:, 0] - vertex_a[:, 0]) * (vertex_b[:, 1] - vertex_a[:, 1]))

    # 3) Choose triangles by area-weighted sampling
    selected_triangle_indices = np.random.choice(len(triangle_indices), size=N, p=triangle_areas / triangle_areas.sum())

    # 4) Generate random barycentric coordinates (Marsaglia trick)
    sqrt_random_1 = np.sqrt(np.random.rand(N, 1))
    random_2 = np.random.rand(N, 1)
    barycentric_u = 1 - sqrt_random_1
    barycentric_v = sqrt_random_1 * (1 - random_2)
    barycentric_w = sqrt_random_1 * random_2

    # 5) Get vertices of selected triangles and compute final positions
    selected_vertex_a = vertex_a[selected_triangle_indices]
    selected_vertex_b = vertex_b[selected_triangle_indices]
    selected_vertex_c = vertex_c[selected_triangle_indices]
    particle_positions = barycentric_u * selected_vertex_a + barycentric_v * selected_vertex_b + barycentric_w * selected_vertex_c
    
    # Ensure we only return 2D positions
    particle_positions = particle_positions[:, :2]
    
    return particle_positions

def sample_star_shape(c, N):
    """
    Generate N evenly spaced (x, y) points along the star-shaped boundary
    defined by equation (5): r(t;c) = c0 + Σ [c_m cos(mt) + c_{m+M} sin(mt)].
    
    Parameters
    ----------
    c : array-like of shape (2M+1,)
        Fourier coefficients [c0, c1..c_M, c_{M+1}..c_{2M}]
    N : int
        Number of sample points along the curve
    
    Returns
    -------
    points : ndarray of shape (N, 2)
        Array of (x, y) coordinates representing the boundary.
    """
    c = np.asarray(c)
    M = (len(c) - 1) // 2  # number of Fourier modes
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Compute r(t)
    r = c[0] * np.ones_like(t)
    for m in range(1, M + 1):
        r += c[m] * np.cos(m * t) + c[m + M] * np.sin(m * t)

    # Convert to Cartesian coordinates
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y])


def create_mesh_from_star_shape(pts, mesh_size=0.1):
    """
    Create a mesh from star shape boundary points.
    
    Parameters
    ----------
    pts : array-like
        Boundary points
    mesh_size : float, optional
        Characteristic mesh size. Larger values = fewer, larger cells.
        Default is 0.1.
    
    Returns
    -------
    mesh : pygmsh mesh object
    """
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(pts, mesh_size=mesh_size)
        mesh = geom.generate_mesh()
    return mesh


def create_cell_list_and_adjacency_lists(mesh):
    """
    Create cell list and edge-to-cells mapping for all triangle cells in the mesh.
    Optimized version using edge-based lookup for O(N) complexity instead of O(N^2).
    
    Parameters
    ----------
    mesh : pygmsh mesh object
        Mesh with triangle cell type
        
    Returns
    -------
    cells : list
        List of cell_triangle objects
    edge_to_cells : dict
        Dictionary mapping Edge objects to lists of cell objects containing each edge.
        Interior edges map to 2 cells, boundary edges map to 1 cell.
    """
    tri_conn = mesh.get_cells_type("triangle")
    pts2d = mesh.points[:, :2]  # Extract 2D coordinates


    cells = []
    for tri in tri_conn:
        verts = pts2d[tri][:, :2]   # shape (3, 2): the triangle's three (x,y) vertices
        cells.append(ct.cell_triangle(verts))
    
    # Build edge-to-cells mapping: each edge maps to list of cell objects containing it
    # Note: edges must be created using vertex coordinates (not indices) to match cell.edges
    edge_to_cells = {}
    for i, tri in enumerate(tri_conn):
        # Get the cell's vertices (as coordinate pairs, not indices)
        verts = cells[i].vertices
        # Get all 3 edges of the triangle using vertex coordinates (matching cell.edges format)
        edges = [
            Edge(verts[0], verts[1]),  # AB edge
            Edge(verts[1], verts[2]),  # BC edge
            Edge(verts[2], verts[0])   # CA edge
        ]

        for edge in edges:
            if edge not in edge_to_cells:
                edge_to_cells[edge] = []
            edge_to_cells[edge].append(cells[i])
    
    #print(f"    Edge-to-cells mapping complete", flush=True)
    return cells, edge_to_cells

def find_nearest_centroid_cell_vectorized(positions, cells):
    """
    Find the nearest centroid cell for multiple positions using vectorized operations.
    Much faster than calling find_nearest_centroid_cell for each position.
    
    Time complexity: O(N × M) where N = number of positions, M = number of cells.
    This is optimal for typical mesh sizes (hundreds of cells).
    
    For very large meshes (thousands of cells), consider using scipy.spatial.cKDTree
    which has O(N × log(M)) query time, but with O(M × log(M)) build overhead.
    KD-trees typically become faster when M > ~1000 cells.
    
    Parameters
    ----------
    positions : array-like of shape (N, 2)
        Particle positions
    cells : list
        List of cell_triangle objects
        
    Returns
    -------
    nearest_cells : list
        List of nearest cell for each position
    """
    if len(positions) == 0:
        return []
    
    positions = np.asarray(positions)
    # Get all cell centers as array
    cell_centers = np.array([cell.center for cell in cells])  # shape: (n_cells, 2)
    
    # Compute distances: (N, n_cells) array
    # positions[:, None, :] is (N, 1, 2), cell_centers[None, :, :] is (1, n_cells, 2)
    # Result is (N, n_cells) - distance from each position to each cell center
    dists_sq = np.sum((positions[:, None, :] - cell_centers[None, :, :])**2, axis=2)
    
    # Find index of nearest cell for each position
    nearest_indices = np.argmin(dists_sq, axis=1)
    
    # Return list of nearest cells
    return [cells[idx] for idx in nearest_indices]

def find_nearest_centroid_cell_kdtree(positions, cells):
    """
    Find the nearest centroid cell for multiple positions using KD-tree.
    
    Parameters
    ----------
    positions : array-like of shape (N, 2)
        Particle positions
    cells : list
        List of cell_triangle objects
        
    Returns
    -------
    nearest_cells : list
        List of nearest cell for each position
    """
    if len(positions) == 0:
        return []
    
    positions = np.asarray(positions)
    # Get all cell centers as array
    cell_centers = np.array([cell.center for cell in cells])  # shape: (n_cells, 2)
    
    tree = cKDTree(cell_centers)
    
    # Query all positions at once: O(N × log(M))
    # Returns (distances, indices) where indices[i] is the index of the nearest cell for positions[i]
    distances, nearest_indices = tree.query(positions)
    
    return [cells[idx] for idx in nearest_indices]

def find_nearest_centroid_cell(position, cells):
    """
    Find the cell with the nearest centroid to the given position.
    (Kept for backward compatibility, but use vectorized version for multiple positions)
    
    Parameters
    ----------
    position : array-like of shape (2,)
        Particle position (x, y)
    cells : list
        List of cell_triangle objects
        
    Returns
    -------
    nearest_cell : cell_triangle
        Cell with nearest centroid
    """
    position = np.asarray(position)
    min_dist_sq = float('inf')
    nearest_cell = None
    
    for cell in cells:
        dist_sq = np.sum((position - cell.center)**2)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_cell = cell
    
    return nearest_cell

def get_boundary_edges(points):
    """
    Get the boundary edges of the arbitrary shape.
    Returns a list of edge segments [(x1,y1,x2,y2), ...]
    """
    edges = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]  # Wrap around to close the shape
        edges.append((p1[0], p1[1], p2[0], p2[1]))
    return edges

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """
    Parameters:
    px, py : float
        x and y coordinates of the point
    x1, y1 : float
        x and y coordinates of the start of the line segment
    x2, y2 : float
        x and y coordinates of the end of  the line segment

    Returns:
    Calculate the distance from a point to a line segment.
    Returns the distance and the closest point on the line segment.
    """
    # Vector from line start to end
    dx = x2 - x1
    dy = y2 - y1
    
    # Vector from line start to point
    px_vec = px - x1
    py_vec = py - y1
    
    # Project point onto line
    line_length_sq = dx * dx + dy * dy
    if line_length_sq < 1e-12:  # Degenerate line
        return np.sqrt(px_vec * px_vec + py_vec * py_vec), (x1, y1)
    
    t = (px_vec * dx + py_vec * dy) / line_length_sq
    t = np.clip(t, 0, 1)  # Clamp to line segment
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    return dist, (closest_x, closest_y)

def get_edge_normal(x1, y1, x2, y2):
    """
    Get the outward normal vector for an edge.
    Returns normalized normal vector pointing outward from the shape.
    """
    # Edge vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Normal vector (perpendicular to edge)
    # For outward normal, we rotate 90 degrees clockwise: (dx, dy) -> (dy, -dx)
    normal_x = dy
    normal_y = -dx
    
    # Normalize
    length = np.sqrt(normal_x**2 + normal_y**2)
    if length > 1e-12:
        normal_x /= length
        normal_y /= length
    
    return normal_x, normal_y

def point_in_polygon(point_x, point_y, polygon_points):
    """
    Parameters:
    point_x, point_y : float
        x and y coordinates of the point
    polygon_points : list of (x, y) tuples
        List of polygon points

    Returns:
    Ray casting algorithm to determine if point is inside polygon.
    Returns True if point is inside polygon, False otherwise.
    """
    num_vertices = len(polygon_points)
    is_inside = False
    
    previous_vertex_index = num_vertices - 1
    for current_vertex_index in range(num_vertices):
        current_x, current_y = polygon_points[current_vertex_index]
        previous_x, previous_y = polygon_points[previous_vertex_index]
        
        if ((current_y > point_y) != (previous_y > point_y)) and (point_x < (previous_x - current_x) * (point_y - current_y) / (previous_y - current_y) + current_x):
            is_inside = not is_inside
        previous_vertex_index = current_vertex_index
    
    return is_inside


def points_in_polygon_vectorized(positions, polygon_points):
    """
    Vectorized point-in-polygon test for multiple points.
    
    Uses ray casting algorithm, vectorized for efficiency.
    
    Parameters
    ----------
    positions : ndarray of shape (N, 2)
        Array of (x, y) coordinates to test
    polygon_points : array-like of shape (M, 2)
        Polygon vertices as (x, y) coordinates
        
    Returns
    -------
    inside_mask : ndarray of shape (N,) dtype bool
        True for points inside polygon, False for outside
    """
    positions = np.asarray(positions)
    polygon_points = np.asarray(polygon_points)
    
    if len(positions) == 0:
        return np.array([], dtype=bool)
    
    num_vertices = len(polygon_points)
    num_points = len(positions)
    
    # Extract coordinates
    px = positions[:, 0]
    py = positions[:, 1]
    
    # Initialize result
    inside = np.zeros(num_points, dtype=bool)
    
    # Vectorized ray casting
    prev_idx = num_vertices - 1
    for curr_idx in range(num_vertices):
        curr_x, curr_y = polygon_points[curr_idx]
        prev_x, prev_y = polygon_points[prev_idx]
        
        # Vectorized condition: point crosses edge?
        # Condition: ((curr_y > py) != (prev_y > py)) AND (px < intersection_x)
        y_cross = (curr_y > py) != (prev_y > py)
        
        # Avoid division by zero
        dy = prev_y - curr_y
        valid = np.abs(dy) > 1e-12
        
        # Compute intersection x-coordinate
        intersection_x = np.where(
            valid,
            (prev_x - curr_x) * (py - curr_y) / dy + curr_x,
            np.inf
        )
        
        # Update inside mask
        mask = y_cross & (px < intersection_x)
        inside[mask] = ~inside[mask]
        
        prev_idx = curr_idx
    
    return inside


def point_to_line_distance_vectorized(positions, edge_segments):
    """
    Vectorized distance from points to line segments.
    
    Parameters
    ----------
    positions : ndarray of shape (N, 2)
        Points to compute distances for
    edge_segments : ndarray of shape (M, 4)
        Edge segments as [x1, y1, x2, y2] for each edge
        
    Returns
    -------
    distances : ndarray of shape (N, M)
        Distance from each point to each edge
    closest_points : ndarray of shape (N, M, 2)
        Closest point on each edge for each point
    """
    positions = np.asarray(positions)
    edge_segments = np.asarray(edge_segments)
    
    if len(positions) == 0 or len(edge_segments) == 0:
        return np.array([]), np.array([])
    
    N = len(positions)
    M = len(edge_segments)
    
    # Extract edge coordinates
    x1 = edge_segments[:, 0]  # (M,)
    y1 = edge_segments[:, 1]  # (M,)
    x2 = edge_segments[:, 2]  # (M,)
    y2 = edge_segments[:, 3]  # (M,)
    
    # Edge vectors
    dx = x2 - x1  # (M,)
    dy = y2 - y1  # (M,)
    line_length_sq = dx**2 + dy**2  # (M,)
    
    # Expand positions for broadcasting
    px = positions[:, 0, None]  # (N, 1)
    py = positions[:, 1, None]  # (N, 1)
    
    # Vectors from edge start to point: (N, M)
    px_vec = px - x1[None, :]  # (N, M)
    py_vec = py - y1[None, :]  # (N, M)
    
    # Project onto edge: t = (px_vec * dx + py_vec * dy) / line_length_sq
    t = (px_vec * dx[None, :] + py_vec * dy[None, :]) / (line_length_sq[None, :] + 1e-12)
    t = np.clip(t, 0, 1)  # (N, M)
    
    # Closest points on edges: (N, M, 2)
    closest_x = x1[None, :, None] + t[:, :, None] * dx[None, :, None]
    closest_y = y1[None, :, None] + t[:, :, None] * dy[None, :, None]
    closest_points = np.stack([closest_x.squeeze(2), closest_y.squeeze(2)], axis=2)  # (N, M, 2)
    
    # Distances: (N, M)
    distances = np.sqrt((px - closest_points[:, :, 0])**2 + 
                       (py - closest_points[:, :, 1])**2)
    
    return distances, closest_points

def triangle_to_follow(position, cell, edge_to_cells):
    """
    Find the triangle to follow for a particle using barycentric coordinates.
    
    Barycentric coordinates:
    - a corresponds to vertex A (vertices[0])
    - b corresponds to vertex B (vertices[1])
    - c corresponds to vertex C (vertices[2])
    
    When a barycentric coordinate is negative, the point is outside on the 
    opposite side of the edge opposite to that vertex:
    - a < 0: point is outside edge BC (opposite to vertex A) -> use edges[1]
    - b < 0: point is outside edge CA (opposite to vertex B) -> use edges[2]
    - c < 0: point is outside edge AB (opposite to vertex C) -> use edges[0]
    
    Parameters
    ----------
    position : array-like of shape (2,)
        Particle position (x, y)
    cell : cell_triangle
        Current cell to check
    edge_to_cells : dict
        Dictionary mapping Edge objects to lists of cell objects
        
    Returns
    -------
    cell_triangle
        The adjacent cell across the edge the particle is outside of, or the current cell if inside
    """
    x, y = position[0], position[1]
    x1, y1 = cell.vertices[0]  # vertex A
    x2, y2 = cell.vertices[1]  # vertex B
    x3, y3 = cell.vertices[2]  # vertex C

    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if abs(denom) < 1e-12:
        # Degenerate triangle, return current cell
        return cell
        
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    c = 1 - a - b
    
    # Map negative barycentric coordinates to the correct edge
    # edges[0] = AB, edges[1] = BC, edges[2] = CA
    if a < 0:
        # Point is outside on the side opposite to vertex A, so it's outside edge BC
        edge = cell.edges[1]  # BC edge
    elif b < 0:
        # Point is outside on the side opposite to vertex B, so it's outside edge CA
        edge = cell.edges[2]  # CA edge
    elif c < 0:
        # Point is outside on the side opposite to vertex C, so it's outside edge AB
        edge = cell.edges[0]  # AB edge
    else:
        # All barycentric coordinates are non-negative, point is inside the triangle
        return cell
    
    # Get cells sharing this edge and return the one that's not the current cell
    candidate_cells = edge_to_cells.get(edge, [])
    for candidate in candidate_cells:
        if candidate is not cell:
            return candidate
    
    # If no adjacent cell found (boundary edge), return current cell
    return cell

def find_containing_cell(position, start_cell, edge_to_cells):
    """
    Iteratively follow triangles using triangle_to_follow until we find a cell where 
    cell.is_inside(position) is satisfied. This is guaranteed to find the correct cell
    for any position within the mesh domain.
    
    Parameters
    ----------
    position : array-like of shape (2,)
        Particle position (x, y)
    start_cell : cell_triangle
        Initial cell to start the search from (typically the nearest centroid cell)
    edge_to_cells : dict
        Dictionary mapping Edge objects to lists of cell objects
        
    Returns
    -------
    cell_triangle
        The cell that contains the position (guaranteed to find it)
    """
    current_cell = start_cell
    
    while True:
        # Check if current cell contains the position
        if current_cell.is_inside(position[0], position[1]):
            return current_cell
        
        # Get the next cell to follow
        next_cell = triangle_to_follow(position, current_cell, edge_to_cells)
        
        # If we're stuck (next_cell is the same as current_cell), we've reached a boundary
        # In this case, the point must be on the boundary, so return the current cell
        if next_cell is current_cell:
            return current_cell
        
        current_cell = next_cell