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


class Tracker:
    def __init__(self, max_distance):
        self.d = max_distance
        self.tracks = []

    def update(self, objects):
        for new_obj in objects:
            idx = self.assign_to_track(new_obj)
            if idx is None:
                self.tracks.append((len(self.tracks), [new_obj]))
            else:
                self.tracks[idx][1].append(new_obj)

    def assign_to_track(self, new_obj):
        for idx, track in self.tracks:
            if self.distance(new_obj, track[-1]) < self.d:
                return idx
        return None

    @staticmethod
    def distance(obj1, obj2):
        return np.sqrt(np.sum((obj1 - obj2)**2))


def pos2int(pos):
    return int(pos[0]), int(pos[1])


def draw_frame(observed, tracks):
    frame = np.ones(FIELD_SIZE, dtype=np.uint8)
    for i, pos in enumerate(observed):
        color = [0, 0, 0]
        color[i] = 255
        cv2.circle(frame, pos2int(pos), 5, color, -1)
    for i, track in tracks:
        color = [0, 0, 0]
        color[i % 3] = 255
        pos0 = track[0]
        for pos in track[1:]:
            cv2.line(frame, pos2int(pos0), pos2int(pos), color, 1)
            pos0 = pos
    cv2.imshow("video", frame)
    key = cv2.waitKey(10)
    if key == ord("q"):
        sys.exit(0)


def main():
    objects = [Obj(FIELD_SIZE[1] / 2, FIELD_SIZE[0] / 2), Obj(100, 200)]
    tracker = Tracker(10)

    while True:
        for obj in objects:
            obj.update()

        observed = [np.array(obj.observe()) for obj in objects]
        tracker.update(observed)
        draw_frame(observed, tracker.tracks)


if __name__ == "__main__":
    main()
