from manim import *
from manim_slides import Slide
import pandas as pd

centrality_df = pd.read_csv("Manim_Chess_CentralityData.csv")

class WAW25(Slide):
    def construct(self):

        # Title Slide
        title = VGroup(
            Text("Analysis and Predictability of Centrality Measures", font_size=48, t2c={"Centrality Measures": BLUE}),
            Text("in Competition Networks", font_size=48, t2c={"Competition Networks": RED}),
            Text("Anthony Bonato, Mariam Walaa", font_size=24, t2w={"[-12:]": BOLD}),
            Text("WAW'25", font_size=20)
        ).arrange(DOWN, center=True).move_to(ORIGIN)
        self.play(FadeIn(title))
        self.next_slide()
        self.clear()

        # Frame: Background
        heading = Text("Applications of Centrality Measures", font_size=46, t2c={"Centrality Measures": BLUE}).to_edge(UP)
        bullet_points = BulletedList(
            "In modern network science, centrality measures are used to study real-life networks --" \
            " urban networks, brain networks, social networks, and more. Recently, they have also been used in machine learning prediction tasks.",
            "Some of the most common centrality measures include degree, closeness, betweenness, and PageRank. Essentially, " \
            "they help identify influential nodes and are often local metrics.",
            font_size=42
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(heading, DOWN, buff=0.5)

        self.play(FadeIn(heading))
        self.next_slide()
        for bullet in bullet_points:
            self.play(FadeIn(bullet))
            self.next_slide()

        self.clear()

        # Frame: Literature Review 1
        heading = Text("Centrality in Competition Networks", font_size=46, t2c={"Competition Networks": RED}).to_edge(UP)
        bullet_points = BulletedList(
            "Dynamic competition networks: detecting alliances and leaders (Bonato 2018)",
            "Centrality in Dynamic Competition Networks (Bonato et al. 2019)",
            "Winner does not take all: Contrasting centrality in adversarial networks (Bonato et al. 2022)",
            font_size=42
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(heading, DOWN, buff=0.5)

        self.play(FadeIn(heading))
        self.next_slide()
        for bullet in bullet_points:
            self.play(FadeIn(bullet))
            self.next_slide()

        self.clear()

        # Frame: Literature Review 2 (non-competition networks)
        heading = Text("Centrality Measures for Predictive Modelling", font_size=46, t2c={"Centrality Measures": BLUE}).to_edge(UP)
        bullet_points = BulletedList(
            "Predicting sentencing outcomes with centrality measures (Morselli et al. 2013)",
            "Predicting epidemic outbreak sizes by centralities (Bucur et al. 2020)",
            "Predicting antipsychotic treatment response in schizophrenia (Liu et al. 2022)",
            font_size=42
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(heading, DOWN, buff=0.5)

        self.play(FadeIn(heading))
        self.next_slide()
        for bullet in bullet_points:
            self.play(FadeIn(bullet))
            self.next_slide()

        self.clear()

        # Frame: Competition Networks
        heading = Text("Properties of Competition Networks", font_size=46, t2c={"Competition Networks": RED}).to_edge(UP)
        bullet_points = BulletedList(
            "Mathematically, a competition network is a directed graph $G = (V, E)$, where each node $u \in V$ represents a competitor, and a directed edge $(u, v) \in E$ exists if $u$ has defeated $v$ at least once in a competition.",
            "On expectation, these competition networks are close to being directed acyclic graphs (DAGs).",
            "These competition networks can capture and model adversarial interactions in, for example, eSports, reality shows, and more.",
            font_size=42
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(heading, DOWN, buff=0.5)

        self.play(FadeIn(heading))
        self.next_slide()
        for bullet in bullet_points:
            self.play(FadeIn(bullet))
            self.next_slide()

        self.clear()

        # Frame: Defining Common Out-Neighbor (CON) Score
        heading = Text("Common Out-Neighbor (CON) Score", font_size=46).to_edge(UP)
        bullet_points = BulletedList(
            "The CON score between $u$ and $v$ (Bonato et al. 2018), denoted as $\\text{CON}(u, v)$, is the number of nodes to which both $u$ and $v$ have directed edges.",
            "$\\text{CON}(u, v)$ counts competitors defeated by both $u$ and $v$, summing up shared victories (i.e., shared out-neighbors) between $u$ and $v$ in the network.",
            "$\\text{CON}_{1}(v)$ counts competitors defeated by $u$ and all other nodes in the network.",
            "Mathematically, it's computed as:",
            font_size=42
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(heading, DOWN, buff=0.5)

        equation = MathTex(
            r"\text{CON}_{1}(v) = \sum_{\substack{u, x \in V(G), \\ u \neq v}} \min(\mathbf{A}[v, x], \mathbf{A}[u, x])"
        ).scale(1).next_to(bullet_points, DOWN, buff=0.7)

        self.play(FadeIn(heading))
        self.next_slide()

        for bullet in bullet_points:
            self.play(FadeIn(bullet))
            self.next_slide()

        self.play(Write(equation))
        self.next_slide()
        self.clear()

        # Frame: Illustration of CON Score Calculation
        heading = Text("Illustration of CON Score Calculation on G", font_size=46).to_edge(UP)
        vertices = ["u", "v", "x", "y", "z"]
        edges = [
            ("u", "x"),
            ("x", "z"),
            ("y", "z"), 
            ("u", "y"),
            ("v", "y"),
        ]
        layout = {
                "u": [-1, 1, 0],
                "v": [-1, -1, 0],
                "x": [1, 1, 0],
                "y": [0, 0, 0],
                "z": [1, -1, 0]
            }
        edge_config = {"stroke_width": 1, "tip_config": {"tip_length": 0.1, "tip_width": 0.1}}
        graph = DiGraph(
            vertices,
            edges,
            layout=layout,
            label_fill_color=WHITE,
            vertex_config={"u": {"fill_color": RED}, "v": {"fill_color": RED}, 
                           "x": {"fill_color": BLUE}, "y": {"fill_color": BLUE},
                           "z": {"fill_color": GREEN}},
            edge_config=edge_config,
            labels=True
        ).scale(2)
        for edge in [("x", "z"), ("y", "z")]:
            graph.edges[edge].set_color(BLUE)
        for edge in [("u", "y"), ("v", "y")]:
            graph.edges[edge].set_color(RED)

        self.play(FadeIn(heading))
        self.next_slide()
        self.play(Create(graph))
        self.next_slide()

        self.play(FadeOut(heading))
        self.next_slide()

        # Frame: Transitive Closure of G
        heading = Text("Transitive Closure of G", font_size=46).to_edge(UP)
        closure_edges = [("u", "z"), ("v", "z")]

        for u, v in closure_edges:
            start = graph.vertices[u].get_center()
            end = graph.vertices[v].get_center()
            arrow = CurvedArrow(
                start, end, 
                color=YELLOW, 
                stroke_width=1.5,
                tip_length=0.1,
                angle=-PI / 2  # adjust curvature direction
            )
            self.play(Create(arrow))
            self.next_slide()

        self.play(FadeIn(heading))
        self.next_slide()

        self.play(FadeOut(heading))
        self.next_slide()

        self.clear()

        # Frame: Use Case for 2nd Order CON Score
        heading = Tex("$u$ and $w$ have a shared competitor in $z$", font_size=46).to_edge(UP)
        edge_config = {"stroke_width": 1, "tip_config": {"tip_length": 0.1, "tip_width": 0.1}}
        vertices.append("w")
        edges.append(("w", "x"))
        layout["w"] = [2, 1, 0]
        graph = DiGraph(
            vertices,
            edges,
            layout=layout,
            label_fill_color=WHITE,
            vertex_config={"u": {"fill_color": GREEN}, "y": {"fill_color": GREEN}, 
                           "x": {"fill_color": GREEN}, "z": {"fill_color": GREEN}, 
                           "w": {"fill_color": GREEN}},
            edge_config=edge_config,
            labels=True
        ).scale(2)

        self.play(Create(graph))
        self.next_slide()

        for edge in [("u", "y"), ("y", "z"), ("x", "z"), ("w", "x")]:
            self.play(graph.edges[edge].animate.set_color(GREEN).set_stroke(width=4))
            self.next_slide()

        self.play(FadeIn(heading))
        self.next_slide()
        
        self.clear()

        # Frame: Second-Order CON Score
        heading = Text("Second-Order CON Score", font_size=46).to_edge(UP)

        bullet_points2 = BulletedList(
            "We extend the CON score to include second-order common out-neighbors (competitors of competitors).",
            "This metric counts shared paths of up to distance two for each pair of nodes.",
            "The new adjacency matrix is:",
            font_size=42
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(heading, DOWN, buff=0.5)

        equation2 = MathTex(r"\mathbf{A}_{2}[i, j] = \mathbf{A}[i, j] + \mathbf{A}^2[i, j]").scale(1).next_to(bullet_points2, DOWN, buff=0.7)

        equation3 = MathTex(
            r"\text{CON}_{2}(v) = \sum_{\substack{u, x \in V(G), \\ u \neq v}} \min(\mathbf{A}_{2}[v, x], \mathbf{A}_{2}[u, x])"
        ).scale(1).next_to(equation2, DOWN, buff=0.7)

        self.play(FadeIn(heading))
        self.next_slide()

        for bullet in bullet_points2:
            self.play(FadeIn(bullet))
            self.next_slide()

        self.play(Write(equation2))
        self.next_slide()

        self.play(Write(equation3))
        self.next_slide()

        self.clear()

        # Frame: CON Score Visualization
        # Graph (a)
        heading = Text("Illustration of 2nd Order CON Score", font_size=46).to_edge(UP)
        edge_config = {"stroke_width": 1, "tip_config": {"tip_length": 0.1, "tip_width": 0.1}}
        vertices_a = ["u", "v", "w"]
        edges_a = [("u", "w"), ("v", "w")]
        graph_a = DiGraph(
            vertices_a, edges_a, 
            layout={"u": [0, 1, 0], "v": [0, 0, 0], "w": [1, 0.5, 0]},
            labels=True, 
            label_fill_color=WHITE,
            vertex_config={"fill_color": BLUE},
            edge_config=edge_config
        ).scale(1.5)

        # Graph (b)
        edge_config = {"stroke_width": 1, "tip_config": {"tip_length": 0.1, "tip_width": 0.1}}
        vertices_b = ["u", "v", "z", "w"]
        edges_b = [("u", "w"), ("v", "z"), ("z", "w")]
        graph_b = DiGraph(
            vertices_b, edges_b, 
            layout={"u": [0, 1, 0], "v": [0, 0, 0], "z": [1, 0, 0], "w": [2, 0.5, 0]},
            labels=True, 
            label_fill_color=WHITE,
            vertex_config={"fill_color": BLUE, "z": {"fill_color": RED}},
            edge_config=edge_config
        ).scale(1.5)

        # Graph (c)
        edge_config = {"stroke_width": 1, "tip_config": {"tip_length": 0.1, "tip_width": 0.1}}
        vertices_c = ["u", "v", "x", "z", "w"]
        edges_c = [("u", "x"), ("x", "w"), ("v", "z"), ("z", "w")]
        graph_c = DiGraph(
            vertices_c, edges_c, 
            layout={"u": [0, 1, 0], "v": [0, 0, 0], "x": [1, 1, 0], "z": [1, 0, 0], "w": [2, 0.5, 0]},
            labels=True, 
            label_fill_color=WHITE,
            vertex_config={"fill_color": BLUE, "x": {"fill_color": RED}, "z": {"fill_color": RED}},
            edge_config=edge_config
        ).scale(1.5)

        graphs = VGroup(graph_a, graph_b, graph_c).arrange(RIGHT, buff=1.5)
        graphs.next_to(heading, DOWN, buff=2)

        self.play(FadeIn(heading))
        self.next_slide()

        graphs = [graph_a, graph_b, graph_c]
        for graph in graphs:
            self.play(Create(graph))
            self.next_slide()
        
        self.clear()

        # Frame: Well-Known Centrality Measures
        heading = Text("Well-Known Centrality Measures", font_size=46).to_edge(UP)
        self.play(Write(heading))
        self.next_slide()

        bp1 = Text("Closeness Centrality: inverse sum of shortest path distances.", font_size=30).next_to(heading, DOWN, buff=0.5).to_edge(LEFT).shift(RIGHT * 1)
        self.play(FadeIn(bp1))
        self.next_slide()

        formula1 = MathTex(r"C(v) = \left( \sum_{u \in V(G) \setminus \{v\}} d(v, u) \right)^{-1}").scale(0.8).next_to(bp1, DOWN).align_to(bp1, LEFT)
        self.play(Write(formula1))
        self.next_slide()

        bp2 = Text("Betweenness Centrality: fraction of shortest paths through a node.", font_size=30).next_to(formula1, DOWN, buff=0.7).align_to(formula1, LEFT)
        self.play(FadeIn(bp2))
        self.next_slide()

        formula2 = MathTex(r"B(v) = \sum_{x, y \in V(G) \setminus \{v\}} \frac{\sigma_{xy}(v)}{\sigma_{xy}}").scale(0.8).next_to(bp2, DOWN).align_to(bp2, LEFT)
        self.play(Write(formula2))
        self.next_slide()

        bp3 = Text("PageRank: computed on the reversed directed graph in competitions.", font_size=30).next_to(formula2, DOWN, buff=0.7).align_to(formula2, LEFT)
        self.play(FadeIn(bp3))
        self.next_slide()

        # Formula 3
        formula3 = MathTex(r"PR(v) = \frac{1-d}{N} + d \sum_{u \in In(v)} \frac{PR(u)}{Out(u)}").scale(0.8).next_to(bp3, DOWN).align_to(bp3, LEFT)
        self.play(Write(formula3))
        self.next_slide()

        self.clear()

        # Frame: Network Datasets
        heading = Text("Network Datasets", font_size=46).to_edge(UP)
        bullets = BulletedList(
            "Voting data from Survivor, a long-running reality competition TV show with 47+ seasons.",
            "Chess.com Titled Tuesday online chess tournaments (100M+ users, 11M daily active).",
            "Professional Dota 2 esports matches (80M+ accounts over the last decade).",
            font_size=42,
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(heading, DOWN, buff=0.5)
        self.play(FadeIn(heading))
        for item in bullets:
            self.play(FadeIn(item))
            self.next_slide()

        self.clear()

        # Frame: 
        heading = Text("Network Dataset Summary Statistics", font_size=46).to_edge(UP)
        table_data = [
            ["Metric", "Survivor", "Chess.com", "Dota 2"],
            ["# Nodes", "806", "933", "493"],
            ["# Edges", "3,662", "16,571", "2,413"],
            ["# Rounds", "12", "18", "8"],
            ["# Competitions", "46", "1", "1"],
            ["Connected", "No", "Yes", "No"],
            ["# WCC", "46", "1", "39"],
            ["# SCC", "90", "152", "199"],
            ["Sparsity", "0.0064", "0.0214", "0.0081"],
            ["Diameter", "3", "4", "10"],
            ["Runtime", "1.5s", "635s", "22s"]
        ]

        table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"stroke_width": 1.5},
            element_to_mobject=Text,
            h_buff=1.0,
            v_buff=0.5
        ).scale(0.5).next_to(heading, DOWN, buff=0.5)

        caption = Text("Low sparsity for all, high runtime for Chess.com, large diameter for Dota 2.",
            font_size=28).next_to(table, DOWN, buff=0.5)

        self.play(FadeIn(heading))
        self.next_slide()

        self.play(FadeIn(table))
        self.next_slide()

        self.play(FadeIn(caption))
        self.next_slide()

        self.clear()

        # Frame: Competition Networks
        #heading = Text("Transitive Triads in Competition Networks", font_size=46).to_edge(UP)
        #bullet_points = BulletedList(
        #    "We define a transitive triad as a set of three nodes $u, v, w$ such that $u$ defeats $v$, $v$ defeats $w$, and $u$ defeats $w$.",
        #    "Transitive triads are important in competition networks as they indicate a clear hierarchy or dominance relationship among competitors.",
        #    font_size=42
        #).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(heading, DOWN, buff=0.5)

        #self.play(FadeIn(heading))
        #self.next_slide()
        #for bullet in bullet_points:
        #    self.play(FadeIn(bullet))
        #    self.next_slide()

        #self.clear()

        # Frame: Chess.com Centrality Measures Plot Matrix
        heading = Text("861 Chess.com Player Rankings, Sorted By Elo", font_size=46).to_edge(UP)
        axes = Axes(
            x_range=[0, len(centrality_df), 100],
            y_range=[0, 1, 0.1],
            x_length=10,
            y_length=5,
            axis_config={"include_numbers": False},
        ).shift(DOWN)

        axes_labels = axes.get_axis_labels(x_label="Player Rank", y_label="Score")

        rank_df_small = centrality_df.sort_values("Elo", ascending=False).reset_index()

        def smooth_series(series, window):
            return series.rolling(window=50, min_periods=1).mean()

        def line_from_column(column, color, y_max):
            smoothed_vals = smooth_series(rank_df_small[column], 50)
            points = [
                axes.c2p(x, min(y, y_max))  # Cap values at y_max to stay inside plot
                for x, y in enumerate(smoothed_vals)
            ]
            return VMobject().set_points_smoothly(points).set_stroke(color, width=3)

        con_line = line_from_column("CON Score", BLUE, 3000)
        pagerank_line = line_from_column("PageRank Centrality", GREEN, 3000)
        elo_line = line_from_column("Elo", YELLOW, 3000)
        outdeg_line = line_from_column("Out-Degree Centrality", RED, 3000)

        legend = VGroup(
            Dot(color=BLUE), Text("CON Score", font_size=28),
            Dot(color=GREEN), Text("PageRank", font_size=28),
            Dot(color=YELLOW), Text("Elo", font_size=28),
            Dot(color=RED), Text("Out-Degree", font_size=28),
        ).arrange_in_grid(rows=2, cols=4, buff=0.4).next_to(axes, DOWN, buff=0.5)

        self.play(FadeIn(heading))
        self.next_slide()
        self.play(Create(axes), Write(axes_labels))
        self.next_slide()
        self.play(Create(con_line))
        self.next_slide()
        self.play(Create(pagerank_line))
        self.next_slide()
        self.play(Create(elo_line))
        self.next_slide()
        self.play(Create(outdeg_line))
        self.next_slide()
        #self.play(FadeIn(legend))
        #self.next_slide()

        self.clear()

        # Frame: Ground Truth Labels
        heading = Text("Ground Truth Labels for Model Training", font_size=46).to_edge(UP)
        bullets = BulletedList(
            "Ground truth labels are essential for supervised learning, providing a benchmark for evaluating centrality-based predictions.",
            "In our datasets, these labels reflect true node rankings, derived from accurate and up-to-date Glicko ratings—independent of sampled game data.",
            "Examples include Survivor contestant outcomes (Sole survivor, runner-up, or voted out), as well as Chess \& Dota 2 (Glicko rating of player or team skill).",
            font_size=42,
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(heading, DOWN, buff=0.5)
        self.play(FadeIn(heading))
        self.next_slide()
        for item in bullets:
            self.play(FadeIn(item))
            self.next_slide()

        self.clear()

        # Frame: Chess.com Ground Truth Correlation
        heading = Text("Chess.com Ground Truth Correlation With Centrality", font_size=46).to_edge(UP)
        bullets = BulletedList(
            "Spearman's R shows out-degree is most correlated with Chess.com rankings, followed by CON score and PageRank.",
            "Closeness has the lowest correlation to known rankings.",
            font_size=42,
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(heading, DOWN, buff=0.5)
        self.play(FadeIn(heading))
        self.next_slide()
        for item in bullets:
            self.play(FadeIn(item))
            self.next_slide()

        table_data = [
            ["Metric", "Correlation", "P-Value"],
            ["Out-Degree", "0.736", "6 × 10⁻¹⁴⁸"],
            ["CON Score", "0.707", "2 × 10⁻¹³¹"],
            ["PageRank", "0.701", "2 × 10⁻¹²⁸"],
            ["Betweenness", "0.664", "2 × 10⁻¹¹⁰"],
            ["Closeness", "0.309", "2 × 10⁻²⁰"]
        ]
        table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"stroke_width": 1.5},
            element_to_mobject=Text,
            h_buff=1.0,
            v_buff=0.5
        ).scale(0.6).next_to(bullets, DOWN, buff=0.5)

        self.play(FadeIn(table))
        self.next_slide()

        self.clear()

        # Frame: Generation of Ground Truth Labels
        heading = Text("Generation of Ground Truth Labels", font_size=46).to_edge(UP)
        bullets = BulletedList(
            "Data is split into training and testing with SMOTE to balance the class distribution by generating synthetic samples for minority classes (Low and High).",
            "Here is a distribution of labels as Low/Medium/High across datasets, where Low is the bottom 10\%, Medium is the middle 10\%-90\%, and High is the top 10\%.",
            font_size=42,
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(heading, DOWN, buff=0.5)
        self.play(FadeIn(heading))
        self.next_slide()
        for item in bullets:
            self.play(FadeIn(item))
            self.next_slide()

        table_data = [
            ["Labels", "Survivor", "Chess.com", "Dota 2"],
            ["Low", "152", "87", "50"],
            ["Medium", "455", "690", "393"],
            ["High", "152", "86", "50"]
        ]

        table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"stroke_width": 1.5},
            element_to_mobject=Text,
            h_buff=1.0,
            v_buff=0.5
        ).scale(0.6).next_to(bullets, DOWN, buff=0.5)

        self.play(FadeIn(table))
        self.next_slide()

        self.clear()

        # Frame: Model Performance
        heading = Text("Outcome Classification - Model Performance", font_size=46).to_edge(UP)
        table_data = [
            ["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1-Score"],
            ["Survivor", "SVM", "0.816", "0.780", "0.807", "0.788"],
            ["Survivor", "RF", "0.750", "0.705", "0.697", "0.700"],
            ["Survivor", "XGB", "0.750", "0.713", "0.695", "0.697"],
            ["Survivor", "GB", "0.728", "0.685", "0.659", "0.666"],
            ["Survivor", "DT", "0.662", "0.607", "0.566", "0.578"],
            ["Chess.com", "SVM", "0.799", "0.266", "0.333", "0.296"],
            ["Chess.com", "RF", "0.811", "0.472", "0.450", "0.452"],
            ["Chess.com", "XGB", "0.803", "0.602", "0.436", "0.506"],
            ["Chess.com", "GB", "0.788", "0.423", "0.429", "0.422"],
            ["Chess.com", "DT", "0.699", "0.446", "0.493", "0.462"],
            ["Dota 2", "SVM", "0.608", "0.570", "0.662", "0.549"],
            ["Dota 2", "RF", "0.770", "0.645", "0.594", "0.606"],
            ["Dota 2", "XGB", "0.791", "0.682", "0.651", "0.666"],
            ["Dota 2", "GB", "0.845", "0.714", "0.663", "0.686"],
            ["Dota 2", "DT", "0.777", "0.615", "0.635", "0.620"]
        ]

        table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"stroke_width": 1.5},
            element_to_mobject=Text,
            h_buff=0.8,
            v_buff=0.4
        ).scale(0.35).next_to(heading, DOWN, buff=0.5)

        caption_text = ("Performance of models at classifying players/teams into one of 3 classes.")
        caption = Text(caption_text, font_size=36).scale(0.6).next_to(table, DOWN, buff=0.5)

        self.play(FadeIn(heading))
        self.next_slide()

        self.play(FadeIn(table))
        self.next_slide()

        self.play(FadeIn(caption))
        self.next_slide()

        self.clear()

        # Frame: Feature Importance
        heading = Text("Dota 2 Gradient Boosting Model Feature Importance", font_size=46).to_edge(UP)
        image = ImageMobject("img/DOTA2_combined_plots.png")  
        image.scale(0.35)  
        image.next_to(heading, DOWN, buff=0.5)  
        self.play(FadeIn(heading))
        self.next_slide()
        self.play(FadeIn(image))
        self.next_slide()
        self.clear()

        # Frame: Conclusion & Discussion
        heading = Text("Conclusion & Discussion", font_size=46).to_edge(UP)
        bullets = BulletedList(
            "We showed that the CON score is a highly effective predictive feature, often outperforming traditional centrality measures like PageRank, closeness, and betweenness.",
            "More can be done! We can extend this work to other kinds of adversarial datasets in biology or economics (e.g. animal dominance or food webs), or on random graphs.",
            "We can also extend the CON score to provide predictive models with more information and globalize the metric.",
            font_size=42,
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(heading, DOWN, buff=0.5)
        self.play(FadeIn(heading))
        self.next_slide()
        for item in bullets:
            self.play(FadeIn(item))
            self.next_slide()
        self.clear()

        # Frame: Links
        heading = Text("Links to arXiv and GitHub", font_size=46).to_edge(UP)
        image1 = ImageMobject("img/arXiv_QR_code.png").scale(0.85)
        image2 = ImageMobject("img/GitHub_QR_Code.png").scale(0.85)

        images = Group(image1, image2).arrange(RIGHT, buff=1.0).next_to(heading, DOWN, buff=0.5)

        self.play(FadeIn(heading))
        self.next_slide()

        self.play(FadeIn(images))
        self.next_slide()

        caption_text = ("Thank you for listening!")
        caption = Text(caption_text, font_size=36).scale(0.6).next_to(table, DOWN, buff=0.5)
        self.play(FadeIn(caption))

        self.clear()

