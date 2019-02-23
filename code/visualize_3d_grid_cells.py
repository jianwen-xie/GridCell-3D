import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
from mayavi import mlab
import matplotlib.pyplot as plt

if __name__ == '__main__':

    weights_path = './training_result/learned_patterns/weights.npy'
    output_path = './training_result/learned_patterns/'


    weights = np.load(weights_path).transpose(3, 0, 1, 2)

    for i in range(len(weights)):
        volume = weights[i]
        [x_dim, y_dim, z_dim] = volume.shape

        fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))

        xslice = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(volume),
                                    plane_orientation='x_axes',
                                    slice_index=x_dim // 2,
                                    colormap='jet'
                                )
        yslice = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(volume),
                                    plane_orientation='y_axes',
                                    slice_index=y_dim // 2,
                                    colormap='jet'
                                )
        zslice = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(volume),
                                plane_orientation='z_axes',
                                slice_index=z_dim // 2,
                                colormap='jet'
                            )

        mlab.outline()


        mlab.view(150, 60, distance=130)

        mlab.savefig(output_path + 'heatmap_' + str(i) + '.png', figure=fig)
        mlab.close()

