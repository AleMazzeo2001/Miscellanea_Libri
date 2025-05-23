from manimlib import *
import numpy as np

# To watch one of these scenes, run the following:
# manimgl example_scenes.py OpeningManimExample
# Use -s to skip to the end and just save the final frame
# Use -w to write the animation to a file
# Use -o to write it to a file and open it once done
# Use -n <number> to skip ahead to the n'th animation of a scene.


class Newton(Scene):
    def algorithm_data(self):
        # es 1
        # fct = lambda x: -(x**3) + x + 3
        # fct_prime = lambda x: -3 * x**2 + 1
        # x0 = 0.5  # 1 e 0.5
        # plt_xaxis = (-4, 4)
        # plt_yaxis = (-8, 8)
        # mol = 1

        # es 2
        # fct = lambda x: x * np.cos(x)
        # fct_prime = lambda x: np.cos(x) - x * np.sin(x)
        # x0 = 0.5  # 1 e 0.5
        # plt_xaxis = (-4, 4)
        # plt_yaxis = (-2, 2)
        # mol = 1

        # es 3
        fct = lambda x: x**4
        fct_prime = lambda x: 4 * x**3
        x0 = 1
        plt_xaxis = (-1.5, 1.5)
        plt_yaxis = (-2, 2)
        mol = 4  # 1 e 4

        num_iter = 10

        plt_wait = 1

        return fct, fct_prime, mol, x0, num_iter, plt_xaxis, plt_yaxis, plt_wait

    def construct(self):

        (
            fct,
            fct_prime,
            mol,
            x0,
            num_iter,
            plt_xaxis,
            plt_yaxis,
            plt_wait,
        ) = self.algorithm_data()

        axes = Axes(plt_xaxis, plt_yaxis)
        axes.add_coordinate_labels()
        self.play(Write(axes, lag_ratio=0.01, run_time=1))

        graph = axes.get_graph(fct, color=GREEN)
        axes.get_graph_label(graph, "f")
        self.play(ShowCreation(graph))

        xk = x0
        tan_line = None
        for num in np.arange(num_iter):

            fk = fct(xk)
            fk_prime = fct_prime(xk)

            xk_dot = Dot(axes.c2p(xk, 0), color=RED)
            xk_text = Tex("x_{" + str(num) + "}", color=RED)
            xk_text.move_to(axes.c2p(xk + 0.2, 0.2))

            self.play(*[FadeIn(o) for o in [xk_dot, xk_text]])

            if tan_line is not None:
                self.play(*[FadeOut(o) for o in [tan_line]])

            xk_line = DashedLine(axes.c2p(xk, 0), axes.c2p(xk, fk))
            self.play(ShowCreation(xk_line))

            fk_dot = Dot(axes.c2p(xk, fk), color=BLUE)
            fk_text = Tex("f(x_{" + str(num) + "})", color=BLUE)
            fk_text.move_to(axes.c2p(xk + 0.4, fk - 0.2))
            self.play(*[FadeIn(o) for o in [fk_dot, fk_text]])

            pos = 1
            tan_line = DashedLine(
                axes.c2p((1 + pos) * xk, mol * fk_prime * pos * xk + fk),
                axes.c2p(-1, mol * fk_prime * (-1 - xk) + fk),
                color=BLUE,
            )
            self.play(ShowCreation(tan_line))

            xk -= mol * fk / fk_prime

            self.play(
                *[FadeOut(o) for o in [xk_dot, xk_text, xk_line, fk_dot, fk_text]]
            )
            self.wait(3)
