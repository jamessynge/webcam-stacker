#!/usr/bin/env python3.6

"""
Initial experiment in stacking a set of captured snapshots.
"""

import glob
import io
import numpy as np
import os
from PIL import Image
import sys


class NotEqualException(Exception):
    pass


def check_equal(a, b):
    if a != b:
        raise NotEqualException(f'{a} != {b}')


class WrongTypeException(Exception):
    pass


def check_types(value, classinfo):
    if not isinstance(value, classinfo):
        raise WrongTypeException(
            f'Value {value} (type {type(value)} is not an instance of {classinfo}')


def jpeg_bytes_to_ndarray(jpeg_bytes):
    #print('len(jpeg_bytes):', len(jpeg_bytes))
    image = np.array(Image.open(io.BytesIO(jpeg_bytes)))
    # print('raw histogram:')
    # print(np.histogram(image, bins=256))

    # print('Image in num:\n', image)
    return image


class ImageStacker(object):
    """docstring for ImageStacker"""
    def __init__(self):
        super(ImageStacker, self).__init__()
        self.stacked_image = None
        self.images = []

    def num_images(self):
        return len(self.images)

    def add(self, image):
        check_types(image, np.ndarray)
        self.images.append(image)
        if self.stacked_image is None:
            self.stacked_image = image.astype('uint32')  # Maybe copy it.
        else:
            check_equal(image.shape, self.stacked_image.shape)
            np.add(self.stacked_image, image, out=self.stacked_image)

    def remove_oldest(self):
        oldest = self.images.pop(0)
        np.add(self.stacked_image, oldest, out=self.stacked_image)
        return oldest

    def scale(self, scale):
        np.multiply(self.stacked_image, scale, out=self.stacked_image)

    def get_byte_image(self):
        # print('stacked_image histogram')
        # print(np.histogram(self.stacked_image, bins=256))

        # Subtract the minimum value so that the values start at zero (black).

        zero_image = self.stacked_image - np.amin(self.stacked_image)

        # print('zero_image histogram')
        # print(np.histogram(zero_image, bins=256))

        # Now divide by the maximum value / 255, bringing the max value to 255.
        zero_image = zero_image.astype('double')
        byte_image = zero_image / (np.amax(zero_image) / 255.0)

        # print('byte_image histogram')
        # print(np.histogram(byte_image, bins=256))

        return byte_image


def main():
    dir = '/tmp/foscam_pictures'
    paths = glob.glob(os.path.join(dir, 'foscam_*.jpg'))
    paths.sort()
    stacker = ImageStacker()
    target_size = 20
    for n, fn in enumerate(paths):
        print('Loading', fn)
        with open(fn, 'rb') as f:
            image = jpeg_bytes_to_ndarray(f.read())
        stacker.add(image)

        while stacker.num_images() > target_size:
            stacker.remove_oldest()

        byte_image = stacker.get_byte_image()

        # print('byte_image histogram')
        # print(np.histogram(byte_image, bins=256))

        byte_image = Image.fromarray(byte_image.astype('uint8'))
        # byte_image.show()

        afn = os.path.join(dir, f'average_foscam_{n:04d}.jpg')
        byte_image.save(afn)
        print('      =>', afn)


if __name__ == '__main__':
    main()
    sys.exit()



    stacked_image = None


    for n in range(48):
        fn = '/home/james/Pictures/foscam-snapshots/snapshot_%d.jpg' % (n+1)
        print('Loading', fn)
        with open(fn, 'rb') as f:
            image = jpeg_bytes_to_ndarray(f.read())
        if stacked_image is None:
            # Convert image to uint32 so there is more range available.
            stacked_image = image.astype('uint32')
        else:
            np.add(stacked_image, image, out=stacked_image)

    print('stacked_image histogram')
    print(np.histogram(stacked_image, bins=256))

    # Subtract the minimum value so that the values start at zero (black).

    zero_image = stacked_image - np.amin(stacked_image)

    print('zero_image histogram')
    print(np.histogram(zero_image, bins=256))

    # Now divide by the maximum value / 255, bringing the max value to 255.

    zero_image = zero_image.astype('double')
    byte_image = zero_image / (np.amax(zero_image) / 255.0)


    print('byte_image histogram')
    print(np.histogram(byte_image, bins=256))

    byte_image = Image.fromarray(byte_image.astype('uint8'))

    byte_image.show()

    byte_image.save('/home/james/Pictures/foscam-snapshots/average-snapshot.jpg')





