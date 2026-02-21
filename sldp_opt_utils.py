import numpy as np

class Rectangle:
    def __init__(self, x_min, y_min, x_max, y_max, parent_count_est=None, depth=0):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.parent_count_est = parent_count_est
        self.depth = depth  
        self.count_est = 0
        self.is_leaf = True

    def split_quad(self):
        """Return childer at the depth + 1."""
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2
        next_depth = self.depth + 1  
        

        c1 = Rectangle(self.x_min, self.y_min, x_mid, y_mid, 
                       parent_count_est=self.count_est, depth=next_depth)
        c2 = Rectangle(x_mid, self.y_min, self.x_max, y_mid, 
                       parent_count_est=self.count_est, depth=next_depth)
        c3 = Rectangle(self.x_min, y_mid, x_mid, self.y_max, 
                       parent_count_est=self.count_est, depth=next_depth)
        c4 = Rectangle(x_mid, y_mid, self.x_max, self.y_max, 
                       parent_count_est=self.count_est, depth=next_depth)
        
        return [c1, c2, c3, c4]

    def __repr__(self):
        return f"Rect(d={self.depth}, est={self.count_est:.1f})"


def run_dp_quadtree_optimized(dataset, epsilon, Q, C, T, delta, bounds, rng):
    """
    Fast version of the private QuadTree construction algorithm.

    Arguments:
        dataset -- np.array (N, 2), point coordinates
        epsilon -- float, privacy budget
        Q       -- int, point count threshold
        C       -- float, scaling constant
        T       -- int, maximum depth
        delta   -- float, reliability parameter (for calculating x_delta)
        bounds  -- tuple (xmin, ymin, xmax, ymax)
        rng     -- np.random.Generator

    Optimization: Uses index passing to avoid filtering the entire dataset 
    at each step. Complexity is reduced from O(N * 4^T) to O(N * T).
    """
    x_min, y_min, x_max, y_max = bounds
    root = Rectangle(x_min, y_min, x_max, y_max)
    x_delta = np.log(1.0 / delta)
    N = dataset.shape[0]
    
    root.count_est = N 
    
    #active_items holds tuples: (cell_object, point_indices_inside)
    active_items = [(root, np.arange(N))] 
    
    final_leaves = []
    t = 0
    
    while len(active_items) > 0 and t < T:
        t += 1
        next_active_items = []
        
        for parent_cell, parent_indices in active_items:
            
            # 1. Retrieve the data subset
            if len(parent_indices) == 0:
                parent_pts = np.empty((0, 2))
            else:
                parent_pts = dataset[parent_indices]

            # 2. Creat children
            children = parent_cell.split_quad()
            
            # 3. Determine which points belong to each child node (compute masks)
            x_mid = (parent_cell.x_min + parent_cell.x_max) / 2
            y_mid = (parent_cell.y_min + parent_cell.y_max) / 2
            
            # Local masks w.r.t. parent_pts
            mask_left = parent_pts[:, 0] < x_mid
            mask_right = ~mask_left
            mask_bottom = parent_pts[:, 1] < y_mid
            mask_top = ~mask_bottom
            
            # Masks for 4 quadrants:
            # c1 (LB), c2 (RB), c3 (LT), c4 (RT)
            masks = [
                mask_left & mask_bottom,
                mask_right & mask_bottom,
                mask_left & mask_top,
                mask_right & mask_top
            ]
            
            # Number of useres in the parent cell
            n_users_in_parent = len(parent_indices)
            
            # Calibrating Delta_t
            n_parent_est = max(0, parent_cell.count_est)
            delta_t = (C / epsilon) * (np.sqrt(2* n_parent_est * x_delta)/C + x_delta)
            
        
            for child, mask in zip(children, masks):
                true_count = np.sum(mask)
                
                # Generate noise
                if n_users_in_parent > 0:
                    noise_sum = np.sum(rng.laplace(0, 1.0/epsilon, size=n_users_in_parent))
                else:
                    noise_sum = 0
                
                child.count_est = true_count + noise_sum
                
                # Desicion criteria
                if child.count_est >= (Q + delta_t):
                    # Split further -> append to queue
                    child_indices = parent_indices[mask] # Пробрасываем индексы дальше
                    next_active_items.append((child, child_indices))
                    child.is_leaf = False 
                else:
                    # Stop -> create a leaf
                    child.is_leaf = True
                    final_leaves.append(child)
        
        active_items = next_active_items

    # Any nodes still active after T steps are forcibly converted to leaf nodes
    for cell, _ in active_items:
        cell.is_leaf = True
        final_leaves.append(cell)
        
    return final_leaves







