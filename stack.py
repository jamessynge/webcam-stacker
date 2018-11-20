#!/usr/bin/env python3.6

"""
Initial experiment in stacking a set of captured snapshots.
"""

import io
import numpy as np
from PIL import Image


def jpeg_bytes_to_ndarray(jpeg_bytes):
    #print('len(jpeg_bytes):', len(jpeg_bytes))
    image = np.array(Image.open(io.BytesIO(jpeg_bytes)))
    print('raw histogram:')
    print(np.histogram(image, bins=256))

    # print('Image in num:\n', image)
    return image

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





