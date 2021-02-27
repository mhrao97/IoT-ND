import numpy

class image_coordinates:
	def __init__(self, midpoint, len_focus):
	    self.mid_point = midpoint
	    self.focal_length = len_focus
	    return
	def get_input_matrix(self):
	    image_coord = np.zeros((3, 3), dtype='float32')
	    x_coord, y_coord = self.mid_point[0], self.mid_point[1]
	    int_x, int_y = int(x_coord), int(y_coord)
	    image_coord[0][0] = self.focal_length
	    image_coord[0][2] = int_x
	    image_coord[1][1] = self.focal_length
	    image_coord[1][2] = int_y
	    image_coord[2][2] = 1
	    return image_coord