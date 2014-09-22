/**************************************************************************
 * Copyright 2010 Pim Schellart. All rights reserved.                     *
 *                                                                        *
 * Redistribution and use in source and binary forms, with or             *
 * without modification, are permitted provided that the following        *
 * conditions are met:                                                    *
 *                                                                        *
 *    1. Redistributions of source code must retain the above             *
 *       copyright notice, this list of conditions and the following      *
 *       disclaimer.                                                      *
 *                                                                        *
 *    2. Redistributions in binary form must reproduce the above          *
 *       copyright notice, this list of conditions and the following      *
 *       disclaimer in the documentation and/or other materials           *
 *       provided with the distribution.                                  *
 *                                                                        *
 * THIS SOFTWARE IS PROVIDED BY PIM SCHELLART ``AS IS'' AND ANY           *
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE      *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR     *
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PIM SCHELLART OR             *
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,    *
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR     *
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY    *
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE      *
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF            *
 * SUCH DAMAGE.                                                           *
 *                                                                        *
 * The views and conclusions contained in the software and documentation  *
 * are those of the authors and should not be interpreted as representing *
 * official policies, either expressed or implied, of Pim Schellart.      *
 **************************************************************************/
#include <Python.h>
#include <boost/python.hpp>
#include "boost/python/numeric.hpp"
#include <iostream>
#include <vector>
#include <math.h>
namespace py = boost::python;

void lombscargle(py::object t, py::object x, py::object w, py::object P)
{
  /** 
   *  Purpose
   *  =======
   * 
   *  Computes the Lomb-Scargle periodogram as developed by Lomb (1976)
   *  and further extended by Scargle (1982) to find, and test the
   *  significance of weak periodic signals with uneven temporal sampling.
   * 
   *  This subroutine calculates the periodogram using a slightly
   *  modified algorithm due to Townsend (2010) which allows the
   *  periodogram to be calculated using only a single pass through
   *  the input samples.
   *  This requires Nw(2Nt+3) trigonometric function evaluations (where
   *  Nw is the number of frequencies and Nt the number of input samples),
   *  giving a factor of ~2 speed increase over the straightforward
   *  implementation.
   * 
   *  Arguments
   *  =========
   * 
   *  t   (input) double precision array, dimension (Nt)
   *      Sample times
   * 
   *  x   (input) double precision array, dimension (Nt)
   *      Measurement values
   * 
   *  w   (input) double precision array, dimension (Nt)
   *      Angular frequencies for output periodogram
   * 
   *  P   (output) double precision array, dimension (Nw)
   *      Lomb-Scargle periodogram
   * 
   * 
   *  Further details
   *  ===============
   * 
   *  P[i] takes a value of A^2*N/4 for a harmonic signal with
   *  frequency w(i).
   **/
  /* length of arrays 
   *  Nt (input) integer
   *      Dimension of input arrays
   * 
   *  Nw (output) integer
   *      Dimension of output array
   */
  Py_Initialize();
  int Nt = py::extract<int>(t.attr("__len__")());
  int Nw = py::extract<int>(P.attr("__len__")());
  /* Local variables */
  int i, j;
  float c, s, xc, xs, cc, ss, cs;
  float theta;
  float tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;
  float term0, term1;
  // make arrays for speed;
  float *T, *W, *X;
  T = (float*) malloc(Nt * sizeof(float));
  X = (float*) malloc(Nt * sizeof(float));
  W = (float*) malloc(Nw * sizeof(float));
  // Put data in c arrays
  for (i = 0; i < Nt; i++)
  {
    T[i] = py::extract<float>(t.attr("__getitem__")(i));
    X[i] = py::extract<float>(x.attr("__getitem__")(i));
  }
  for( i = 0; i < Nw; i++)
  {
    W[i] = py::extract<float>(w.attr("__getitem__")(i));
  }
  for (i = 0; i < Nw; i++)
  {
    xc = 0.;
    xs = 0.;
    cc = 0.;
    ss = 0.;
    cs = 0.;

    for (j = 0; j < Nt; j++)
    {
      theta = W[i] * T[j]; 
      c = cos(theta);
      s = sin(theta);

      xc += X[j] * c;
      xs += X[j] * s;
      cc += c * c;
      ss += s * s;
      cs += c * s;

    }

    tau = atan(2 * cs / (cc - ss)) / (2 *  W[i]);
    theta = W[i] * tau;
    c_tau = cos(theta);
    s_tau = sin(theta);
    c_tau2 = c_tau * c_tau;
    s_tau2 = s_tau * s_tau;
    cs_tau = 2 * c_tau * s_tau;

    term0 = c_tau * xc + s_tau * xs;
    term1 = c_tau * xs - s_tau * xc;
    P[i] = 0.5 * (((term0 * term0) / \
                   (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
                  ((term1 * term1) / \
                   (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));

  }
}

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
using namespace boost::python;
 
BOOST_PYTHON_MODULE(cpp_imp)
{
    def("lomb_cpp", lombscargle);
}
