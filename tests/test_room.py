import unittest
from frads import room


class TestRoom(unittest.TestCase):
    def test_make_room(self):
        width = 3
        depth = 5
        floor_floor = 4
        floor_ceiling = 3
        windows = [
            [1., 1., 1., 1.],
        ]
        thickness = 0.1
        aroom = room.create_south_facing_room(
            width=width,
            depth=depth,
            floor_floor=floor_floor,
            floor_ceiling=floor_ceiling,
            wpd=windows,
            swall_thickness=thickness,
        )

if __name__ == "__main__":
    unittest.main()
