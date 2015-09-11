import pyopencl as cl
import numpy as np

class GPU:
    def __init__(self):
        platform = cl.get_platforms()[0]    # Select the first platform [0]
        device = platform.get_devices()[0]  # Select the first device on this platform [0]
        self.ctx = cl.Context([device])      # Create a context with your device
        self.queue = cl.CommandQueue(self.ctx)

    def load_program(self):
        f = open('pynoise/kernels.cl', 'r')
        fstr = "".join(f.readlines())

        self.program = cl.Program(self.ctx, fstr).build()

    def gradient_noise(self, x, y, z, seed, quality):
        mf = cl.mem_flags

        x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
        y_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
        z_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z)
        dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, x.nbytes)

        self.program.gradient_coherent_noise_3d(self.queue, x.shape, None,
        x_buf, y_buf, z_buf, dest_buf, seed, quality)


        rv = np.empty_like(x)

        cl.enqueue_read_buffer(self.queue, dest_buf, rv).wait()

        return rv
