#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <vector>

// ========================= Global Settings =========================
using scalar_type = double;

template <int N>
using vecN = Eigen::Vector<scalar_type, N>;

using ivec = Eigen::Vector<int, 2>;
using vec = Eigen::Vector<scalar_type, 2>;

constexpr unsigned int FLAG_SOLID = 1u;

ivec idx2dir(int idx)
{
  assert(idx < 9);
  int x = idx % 3;
  int y = idx / 3;

  return { x - 1, y - 1 };
}
int dir2idx(ivec dir)
{
  return dir.x() + 1 + (dir.y() + 1) * 3;
}
scalar_type weight(ivec dir)
{
  int manhat = std::abs(dir.x()) + std::abs(dir.y());
  assert(manhat < 3);
  if (manhat == 0)
  {
    return 4.0 / 9.0;
  }
  if (manhat == 1)
  {
    return 1.0 / 9.0;
  }
  if (manhat == 2)
  {
    return 1.0 / 36.0;
  }
  return 0;
}
scalar_type weight(int idx)
{
  return weight(idx2dir(idx));
}

struct CylinderWakeLBM
{
  int width, height;

  scalar_type dx, dt;

  scalar_type unit_vel;

  // nondimensionalized
  scalar_type tau;

  // nondimensionalized
  scalar_type u0;

  std::vector<vecN<9>> f, ftemp;
  std::vector<vec> velocity;
  std::vector<Eigen::Vector<float, 2>> vel_dim;
  std::vector<scalar_type> density;
  std::vector<float> fdensity;
  std::vector<unsigned int> flag;

  template <typename T>
  T& at(std::vector<T>& v, int x, int y)
  {
    return v[x + y * width];
  }
  template <typename T>
  T& at(std::vector<T>& v, ivec p)
  {
    return at(v, p.x(), p.y());
  }

  bool valid(int x, int y) const
  {
    return x >= 0 && x < width && y >= 0 && y < width;
  }
  bool valid(ivec v) const
  {
    return valid(v.x(), v.y());
  }

  void init(int w, int h, scalar_type nu, scalar_type inlet_vel, scalar_type dt)
  {
    width = w;
    height = h;
    dx = 10.0 / (width - 1);
    this->dt = dt;

    const int wh = width * height;
    f.resize(wh);
    ftemp.resize(wh);
    velocity.resize(wh);
    density.resize(wh);
    vel_dim.resize(wh);
    fdensity.resize(wh);
    flag.resize(wh, 0);

    // nondim
    nu /= (dx * dx / dt);
    unit_vel = dx / dt;
    u0 = inlet_vel / unit_vel;

    tau = (6 * nu + 1) / 2.0;

    std::cout << "W: " << width << "\n";
    std::cout << "H: " << height << "\n";
    std::cout << "dx: " << dx << "\n";
    std::cout << "dt: " << dt << "\n";
    std::cout << "Inlet: " << inlet_vel << "\n";
    std::cout << "Inlet(nondim): " << u0 << "\n";
    std::cout << "tau: " << tau << "\n";

    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        at(velocity, x, y) = vec(u0, 0);
        at(density, x, y) = 1;

        scalar_type fx = x * dx;
        scalar_type fy = y * dx;
        fx -= 2.5;
        fy -= 2.5;
        if (fx * fx + fy * fy < 0.5 * 0.5)
        {
          at(flag, x, y) |= FLAG_SOLID;
          at(velocity, x, y) = vec(0, 0);
        }

        at(f, x, y) = equilibrium(at(velocity, x, y), at(density, x, y));
      }
    }
  }

  void prepare_print()
  {
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        at(vel_dim, x, y) = (at(velocity, x, y) * unit_vel).cast<float>();
        at(fdensity, x, y) = (float)at(density, x, y);
      }
    }
  }

  void step()
  {
#define INVOKE(name)                   \
  for (int y = 0; y < height; ++y)     \
  {                                    \
    for (int x = 0; x < width; ++x)    \
    {                                  \
      if (at(flag, x, y) & FLAG_SOLID) \
        continue;                      \
      name(x, y);                      \
    }                                  \
  }
    INVOKE(streaming);
    boundary_condition();
    INVOKE(macroscopic);
    INVOKE(equilibrium);
    INVOKE(collision);
  }
  vecN<9> equilibrium(vec vel, scalar_type dens)
  {
    vecN<9> ret;
    for (int i = 0; i < 9; ++i)
    {
      vec dir = idx2dir(i).cast<scalar_type>();
      scalar_type edotu = dir.dot(vel);

      scalar_type feq = weight(i) * dens
                        * (1.0 + 3.0 * edotu + 4.5 * edotu * edotu
                           - 1.5 * vel.squaredNorm());
      ret[i] = feq;
    }
    return ret;
  }
  void streaming(int x, int y)
  {
    // set (x,y)'s dir direction
    for (int dir = 0; dir < 9; ++dir)
    {
      ivec fromdir = -idx2dir(dir);
      ivec adjpos = { x, y };
      adjpos -= idx2dir(dir);

      // periodic boundary condition for upper and lower
      if (adjpos.y() == -1)
      {
        adjpos.y() = height - 1;
      }
      if (adjpos.y() == height)
      {
        adjpos.y() = 0;
      }

      if ((valid(adjpos) == false) || (at(flag, adjpos) & FLAG_SOLID))
      {
        continue;
      }

      at(ftemp, x, y)[dir] = at(f, adjpos)[dir];
    }
  }
  void boundary_condition()
  {
    // left inlet ( vel - dirichlet, dens - neumann )
    for (int y = 0; y < height; ++y)
    {
      scalar_type dens = at(density, 1, y);
      vec vel = { u0, 0 };
      at(velocity, 0, y) = vel;
      at(density, 0, y) = dens;
      at(ftemp, 0, y) = equilibrium(vel, dens);
    }
    // right outlet ( vel - neumann, dens - dirichlet )
    for (int y = 0; y < height; ++y)
    {
      vec vel = at(velocity, width - 2, y);
      scalar_type dens = 1.0;
      at(velocity, width - 1, y) = vel;
      at(density, width - 1, y) = dens;
      at(ftemp, width - 1, y) = equilibrium(vel, dens);
    }

    // HBB
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        for (int dir = 0; dir < 9; ++dir)
        {
          ivec adjpos = { x, y };
          adjpos += idx2dir(dir);
          if (valid(adjpos) == false)
          {
            continue;
          }

          if (at(flag, adjpos) & FLAG_SOLID)
          {
            at(ftemp, x, y)[dir2idx(-idx2dir(dir))] = at(ftemp, x, y)[dir];
          }
        }
      }
    }
  }
  void macroscopic(int x, int y)
  {
    scalar_type dens = 0;
    vec vel = { 0, 0 };
    for (int i = 0; i < 9; ++i)
    {
      vec dir = idx2dir(i).cast<scalar_type>();

      dens += at(ftemp, x, y)[i];
      vel += dir * at(ftemp, x, y)[i];
    }
    vel /= dens;
    at(density, x, y) = dens;
    at(velocity, x, y) = vel;
  }
  void equilibrium(int x, int y)
  {
    const vec vel = at(velocity, x, y);
    const scalar_type dens = at(density, x, y);
    at(f, x, y) = equilibrium(vel, dens);
  }

  void collision(int x, int y)
  {
    // collision f(feq), ftemp(fstar) -> f
    auto fstar = at(ftemp, x, y);
    auto feq = at(f, x, y);

    fstar = fstar - (fstar - feq) / tau;
    at(f, x, y) = fstar;
  }
};

int main(int argc, char** argv)
{
  std::vector<int> Res = { 5, 20, 40, 60, 100, 150, 200 };

  for (int re : Res)
  {
    std::cout << "Calculating Reynolds: " << re << "\n";
    std::ofstream stream(std::string("re") + std::to_string(re) + ".dat",
                         std::ios::binary);
    CylinderWakeLBM cylinder;
    scalar_type nu = 1.0 / (double)re;
    cylinder.init(512, 256, nu, 1.0, 0.002);

    // dt = 0.002
    // dx = 10 / 511
    // simulation end: 30
    // simulation iteration: 20 / 0.002 = 15000
    // plot_interval: 0.2
    // plot iter: 0.2 / 0.002 = 100

    for (int i = 0; i < 15000; ++i)
    {
      if (i % 100 == 0)
      {
        std::cout << i << "\n";
        cylinder.prepare_print();
        stream.write((char*)cylinder.vel_dim.data(),
                     sizeof(float) * cylinder.width * cylinder.height * 2);
      }
      cylinder.step();
    }
  }

  return 0;
}