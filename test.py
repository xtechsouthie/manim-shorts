from manim import *
import numpy as np

class Segment1(ThreeDScene):
    def construct(self):
        self.camera.background_color = "#000000"
        
        # Setup 3D axes and title
        axes_3d = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-1, 5, 1],
            height=6,
            width=6,
            depth=4,
            axis_config={"color": WHITE},
            tips=False
        )
        
        title = MathTex("z = f(x,y)", color=YELLOW).scale(1.2)
        title.to_corner(UP + LEFT)
        
        # 2D Reminder on left side
        number_plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-1, 5, 1],
            height=4,
            width=4,
            background_line_style={"stroke_color": BLUE, "stroke_width": 0.5},
            axis_config={"color": WHITE}
        )
        number_plane.shift(LEFT * 5 + DOWN * 0.5)
        
        def f1(t):
            return 0.5 * t**2
        
        param_curve = ParametricFunction(
            lambda t: [t, f1(t), 0],
            t_range=[-2.5, 2.5],
            color=BLUE
        )
        param_curve.scale(0.8)
        param_curve.shift(LEFT * 5 + DOWN * 0.5)
        
        derivative_text = Text("derivative = slope", color=WHITE, font_size=24)
        derivative_text.next_to(number_plane, DOWN, buff=0.3)
        
        # Tangent line at t=1
        t0 = 1
        slope = t0
        p_start = np.array([t0 - 1, f1(t0 - 1), 0])
        p_end = np.array([t0 + 1, f1(t0 + 1), 0])
        tangent_line = Line(p_start, p_end, color=ORANGE, stroke_width=3)
        tangent_line.scale(0.8)
        tangent_line.shift(LEFT * 5 + DOWN * 0.5)
        
        # Animations: 2D setup
        self.play(FadeIn(number_plane), run_time=0.5)
        self.play(Create(param_curve), run_time=0.6)
        self.play(Write(derivative_text), run_time=0.5)
        self.play(Create(tangent_line), run_time=0.4)
        
        self.wait(1)
        
        # Transition to 3D
        self.play(
            FadeOut(number_plane),
            FadeOut(param_curve),
            FadeOut(derivative_text),
            FadeOut(tangent_line),
            run_time=1
        )
        
        self.play(FadeIn(axes_3d), run_time=0.8)
        self.play(Write(title), run_time=0.5)
        
        self.wait(0.5)
        
        # Create 3D surface
        def f(u, v):
            return 0.2 * (u**2 + v**2)
        
        surface = Surface(
            lambda u, v: [u, v, f(u, v)],
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(25, 25),
            color=BLUE,
            opacity=0.8
        )
        
        self.play(Create(surface), run_time=2.5)
        
        # Set up 3D camera view
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
        
        self.wait(1)
        
        # Pick a point on surface
        x0, y0 = 1, 1
        z0 = f(x0, y0)
        point_3d = np.array([x0, y0, z0])
        
        dot = Dot3D(point_3d, color=BLUE, radius=0.12)
        self.play(FadeIn(dot), run_time=0.6)
        
        self.wait(0.5)
        
        # Create tangent plane
        grad_x = 0.4 * x0
        grad_y = 0.4 * y0
        normal = np.array([grad_x, grad_y, -1])
        normal = normal / np.linalg.norm(normal)
        
        plane_size = 2
        u_vec = np.array([1, 0, 0.4 * x0])
        u_vec = u_vec / np.linalg.norm(u_vec)
        v_vec = np.cross(normal, u_vec)
        v_vec = v_vec / np.linalg.norm(v_vec)
        
        corners = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                corner = point_3d + i * plane_size * u_vec + j * plane_size * v_vec
                corners.append(corner)
        
        plane_vertices = [corners[0], corners[1], corners[3], corners[2]]
        tangent_plane = Polygon(*plane_vertices, color=YELLOW, fill_opacity=0.6, stroke_width=2)
        
        self.play(Create(tangent_plane), run_time=1.5)
        
        self.wait(1)
        
        # Label tangent plane
        plane_label = MathTex("\\text{tangent plane}", color=YELLOW, font_size=32)
        plane_label.next_to(dot, UP + RIGHT, buff=0.5)
        
        self.play(Write(plane_label), run_time=0.8)
        
        self.wait(1)
        
        # Rotate camera slowly
        self.play(
            self.camera.animate.set_euler_angles(
                phi=50 * DEGREES,
                theta=60 * DEGREES
            ),
            run_time=3
        )
        
        self.wait(2)
        
        # Highlight plane edge
        self.play(
            tangent_plane.animate.set_stroke(color=WHITE, width=4),
            run_time=1
        )
        
        self.wait(4)