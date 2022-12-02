import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

FIELD_SIZE = (480, 640, 3)


class Obj:
    def __init__(self, x0, y0):
        self.pos = [x0, y0]
        self.speed = np.random.random((1))
        self.direction = np.random.random((1)) * 2 * np.pi

    def update(self):
        self.speed = np.clip(self.speed + np.random.random((1)) - .5, 0, 1)
        self.direction += np.random.random((1)) - .5
        self.pos[0] += np.sin(self.direction) * self.speed
        self.pos[1] += np.cos(self.direction) * self.speed

    def observe(self):
        return self.pos


def draw_frame(observed):
    frame = np.ones(FIELD_SIZE, dtype=np.uint8)
    for i, pos in enumerate(observed):
        color = [0, 0, 0]
        color[i] = 255
        cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, color, -1)
    cv2.imshow("video", frame)
    key = cv2.waitKey(10)
    if key == ord("q"):
        sys.exit(0)


def main():
    objects = [Obj(FIELD_SIZE[1] / 2, FIELD_SIZE[0] / 2), Obj(100, 200)]

    while True:
        for obj in objects:
            obj.update()

        observed = [obj.observe() for obj in objects]
        draw_frame(observed)


if __name__ == "__main__":
    main()
