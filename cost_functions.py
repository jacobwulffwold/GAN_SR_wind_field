import torch
import pyvista as pv
from download_data import download_and_combine, slice_data
from datetime import date
from pyvista import examples
import numpy as np


def plot_wind_field_and_gradient(gradient_mesh, z_plot_scale):
    new_grid = pv.StructuredGrid(
        gradient_mesh.x, gradient_mesh.y, z_plot_scale * gradient_mesh.z
    )
    new_grid.point_data["wind_field"] = gradient_mesh["wind_field"]
    glyphs = new_grid.glyph(orient="wind_field", factor=70, geom=pv.Arrow())
    grad_u = gradient_mesh["gradient"][:, :3]
    grad_v = gradient_mesh["gradient"][:, 3:6]
    grad_w = gradient_mesh["gradient"][:, 6:]
    new_grid.point_data["grad_u"] = grad_u
    new_grid.point_data["grad_v"] = grad_v
    glyphs_grad_u = new_grid.glyph(orient="grad_u", factor=5000, geom=pv.Arrow())
    glyphs_grad_v = new_grid.glyph(orient="grad_v", factor=5000, geom=pv.Arrow())
    new_grid.point_data["du_dx"] = gradient_mesh["gradient"][:, 0]
    new_grid.point_data["du_dy"] = gradient_mesh["gradient"][:, 1]
    new_grid.point_data["dv_dx"] = gradient_mesh["gradient"][:, 3]
    new_grid.point_data["dv_dy"] = gradient_mesh["gradient"][:, 4]

    new_grid.point_data["u"] = gradient_mesh["wind_field"][:, 0]
    new_grid.point_data["v"] = gradient_mesh["wind_field"][:, 1]
    new_grid.point_data["w"] = gradient_mesh["wind_field"][:, 2]

    pl = pv.Plotter(shape="1|2")
    pl.subplot(0)
    pl.add_mesh(glyphs, cmap="coolwarm")
    pl.add_text("Wind Field")
    pl.subplot(1)

    pl.subplot(1)
    pl.add_mesh(glyphs_grad_u, cmap="coolwarm")
    pl.add_text("Gradient of u")
    pl.subplot(1)

    pl.subplot(2)
    pl.add_mesh(glyphs_grad_v, cmap="coolwarm")
    pl.add_text("Gradient of v")
    pl.subplot(1)

    # pl.add_mesh(gradient_mesh, scalars="gradient", cmap='coolwarm', opacity=0.5)
    # pl.add_text("Gradient")
    # # pl.subplot(0,1)
    # # pl.add_mesh(gradient_mesh, scalars="divergence", cmap='coolwarm')
    # # pl.add_text("Divergence")
    # pl.subplot(2)
    # pl.add_mesh(gradient_mesh, scalars="divergence", cmap='coolwarm', opacity=0.5)
    # pl.add_text("DIV")
    # pl.subplot(1,1)
    # pl.add_mesh(mesh, scalars="dv_dx", cmap='coolwarm')
    # pl.add_text("dv_dx")
    pl.show()


def gradient_of_irregular_grid(X, Y, Z, u, v, w):
    mesh = pv.StructuredGrid(X, Y, Z)
    mesh.point_data["wind_field"] = np.asarray(
        [u.flatten(), v.flatten(), w.flatten()]
    ).T
    gradient_mesh = mesh.compute_derivative(scalars="wind_field", divergence=True)

    return gradient_mesh["gradient"], gradient_mesh["divergence"]


if __name__ == "__main__":
    data_code = "simra_BESSAKER_"
    start_date = date(2018, 4, 1)  # 1,2
    end_date = date(2018, 4, 2)  #

    time, terrain, x, y, z, u, v, w, theta, tke, td, pressure = download_and_combine(
        data_code, start_date, end_date
    )

    z_plot_scale = 5
    time_index = 1
    X_DICT = {"start": 4, "max": -115, "step": 1}
    Z_DICT = {"start": 1, "max": 5, "step": 1}

    terrain, x, y, X, Y, z, u, v, w, pressure = slice_data(
        terrain, x, y, z, u, v, w, pressure, X_DICT, Z_DICT
    )

    gradient_mesh = gradient_of_irregular_grid(
        X, Y, z[time_index], u[time_index], v[time_index], w[time_index]
    )

    plot_wind_field_and_gradient(gradient_mesh, z_plot_scale)
