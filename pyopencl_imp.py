import pyopencl as cl
import numpy as np

'''OpenCl version'''
lomb_txt = '''__kernel void lombscargle(__global float2 *x,__global float2 *y,__global float2 *f,__global float2 *P, int Nt, int Nw)
{
  /* Local variables */
  int i, j;
  float2 c, s, xc, xs, cc, ss, cs;
  float2 tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;
  float2 term0, term1;


  for(i = 0; i < Nw; i++)
  {
    xc = 0.;
    xs = 0.;
    cc = 0.;
    ss = 0.;
    cs = 0.;
    for(j = 0; j < Nt; j++)
    {
      c = cos(f[i] * x[j]);
      s = sin(f[i] * x[j]);

      xc += y[j] * c;
      xs += y[j] * s;
      cc += c * c;
      ss += s * s;
      cs += c * s;

     }

    tau = atan(2 * cs / (cc - ss)) / (2 * f[i]);
    c_tau = cos(f[i] * tau);
    s_tau = sin(f[i] * tau);
    c_tau2 = c_tau * c_tau;
    s_tau2 = s_tau * s_tau;
    cs_tau = 2 * c_tau * s_tau;

    term0 = c_tau * xc + s_tau * xs;
    term1 = c_tau * xs - s_tau * xc;
    P[i] = 0.5 * (((term0 * term0) / (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + ((term1 * term1) / (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));
   }
  }'''

def lombscarge_opencl(x, y, f):
    # start up gpu
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    # make max arrays
    Nx, Nf = np.int32(x.shape[0]), np.int32(f.shape[0])
    # send data to card
    x_g = cl.Buffer(ctx, mf.READ_ONLY, x.nbytes)
    y_g = cl.Buffer(ctx, mf.READ_ONLY, y.nbytes)
    f_g = cl.Buffer(ctx, mf.READ_ONLY, f.nbytes)
    #Nx_g = cl.Buffer(ctx, mf.READ_ONLY, Nx.nbytes)
    #Nf_g = cl.Buffer(ctx, mf.READ_ONLY, Nf.nbytes)
    # make output
    pgram = np.empty(f.shape)
    pgram_g = cl.Buffer(ctx, mf.WRITE_ONLY, pgram.nbytes)
    prg = cl.Program(ctx, lomb_txt)

    try:
        prg.build()
    except:
        print("Error:")
        print(prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
        raise

    prg.lombscargle(queue, pgram.shape, None, *(x_g, y_g, f_g, pgram_g, Nx, Nf))
    cl.enqueue_read_buffer(queue, pgram_g, pgram)

    return pgram
if __name__ == '__main__':
    from benchmarks import short_example
    from pyopencl_imp import *
    short_example.scipy_example(lombscarge_opencl)
