import pygame
import sys
import time
import random
import heapq
from collections import deque

# --- CONFIGURACIÓN DE PANTALLA Y MAPA (70 x 50) ---
COLS, ROWS = 70, 50
CELL_SIZE = 16
GRID_WIDTH = COLS * CELL_SIZE            # 1120 px
PANEL_WIDTH = 340                        # 340 px
SCREEN_WIDTH = GRID_WIDTH + PANEL_WIDTH  # 1460 px
SCREEN_HEIGHT = ROWS * CELL_SIZE         # 800 px

# --- PALETA DE COLORES (Dark Mode Neón) ---
COLOR_BG_PANEL = (20, 22, 28)       # Fondo del panel lateral
COLOR_GRID_BG = (10, 12, 15)        # Fondo del mapa (casi negro)
COLOR_GRID_LINE = (35, 40, 50)      # Líneas de la cuadrícula
COLOR_WALL = (220, 225, 230)        # Muros (blanco tiza)
COLOR_START = (0, 190, 255)         # Punto de inicio (cian)
COLOR_GOAL = (255, 80, 80)          # Meta (rojo coral)
COLOR_PATH = (0, 255, 128)          # Ruta óptima (verde neón)
COLOR_EXPLORED = (30, 45, 65)       # Región conexa navegable (azul noche)
COLOR_NPC = (255, 200, 0)           # Agentes NPC (amarillo dorado)
COLOR_SEPARATOR = (45, 50, 60)      # Línea divisoria entre panel y mapa

# Colores tipográficos
TEXT_TITLE = (0, 190, 255)
TEXT_SUB = (150, 160, 170)
TEXT_HEADER = (255, 200, 0)
TEXT_BODY = (220, 220, 220)
TEXT_HIGHLIGHT = (0, 255, 128)

# --- ALGORITMOS DE BÚSQUEDA Y GRAFOS ---

def get_neighbors(node, grid):
    r, c = node
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            if grid[nr][nc] == 0:
                neighbors.append((nr, nc))
    return neighbors

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def run_bfs(start, goal, grid):
    queue = deque([start])
    came_from = {}
    visited = {start}
    explored_nodes = set()

    while queue:
        current = queue.popleft()
        explored_nodes.add(current)
        if current == goal:
            return reconstruct_path(came_from, current), explored_nodes
        for neighbor in get_neighbors(current, grid):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
    return [], explored_nodes

def run_dfs(start, goal, grid):
    stack = [start]
    came_from = {}
    visited = {start}
    explored_nodes = set()

    while stack:
        current = stack.pop()
        explored_nodes.add(current)
        if current == goal:
            return reconstruct_path(came_from, current), explored_nodes
        for neighbor in get_neighbors(current, grid):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
    return [], explored_nodes

def run_a_star(start, goal, grid):
    counter = 0
    open_heap = []
    heapq.heappush(open_heap, (0, counter, start))
    came_from = {}
    g_score = {start: 0}
    explored_nodes = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        explored_nodes.add(current)
        if current == goal:
            return reconstruct_path(came_from, current), explored_nodes
        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, neighbor))
    return [], explored_nodes

def run_dijkstra(start, goal, grid):
    counter = 0
    open_heap = []
    heapq.heappush(open_heap, (0, counter, start))
    came_from = {}
    dist = {start: 0}
    explored_nodes = set()

    while open_heap:
        d, _, current = heapq.heappop(open_heap)
        explored_nodes.add(current)
        if current == goal:
            return reconstruct_path(came_from, current), explored_nodes
        if d > dist.get(current, float('inf')):
            continue
        for neighbor in get_neighbors(current, grid):
            new_d = d + 1
            if neighbor not in dist or new_d < dist[neighbor]:
                dist[neighbor] = new_d
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(open_heap, (new_d, counter, neighbor))
    return [], explored_nodes

def run_greedy(start, goal, grid):
    counter = 0
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))
    came_from = {}
    visited = {start}
    explored_nodes = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        explored_nodes.add(current)
        if current == goal:
            return reconstruct_path(came_from, current), explored_nodes
        for neighbor in get_neighbors(current, grid):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(open_heap, (heuristic(neighbor, goal), counter, neighbor))
    return [], explored_nodes

def get_connected_component(start, grid):
    component = set()
    queue = deque([start])
    visited = {start}
    while queue:
        current = queue.popleft()
        component.add(current)
        for neighbor in get_neighbors(current, grid):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return component

# --- CLASE PRINCIPAL ---

class PathfindingVisualizer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Sistema de Navegación para NPCs - Pathfinding")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fuentes
        self.font_title = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font_subtitle = pygame.font.SysFont("Segoe UI", 12)
        self.font_header = pygame.font.SysFont("Segoe UI", 14, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI", 12)
        self.font_mono = pygame.font.SysFont("Consolas", 11)

        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.start = (3, 3)
        self.goal = (45, 66)  # (fila, col) -> (y: 45, x: 66)
        self.npcs = []

        self.algoritmo_id = 3
        self.nombres_algoritmos = {
            1: "BFS - Breadth First Search",
            2: "DFS - Depth First Search",
            3: "A* - Algoritmo Estrella",
            4: "Dijkstra - Caminos Mínimos",
            5: "Greedy - Búsqueda Voraz"
        }
        self.mostrar_componentes = True
        self.path = []
        self.explored = set()
        self.tiempo_ms = 0.0
        self.historial = []

        self.generar_muros_defecto()
        self.ejecutar_algoritmo()

    def generar_muros_defecto(self):
        for c in range(6, 30): self.grid[8][c] = 1
        for r in range(8, 42): self.grid[r][20] = 1
        for r in range(4, 30): self.grid[r][40] = 1
        for c in range(35, 60): self.grid[22][c] = 1
        for r in range(8, 35): self.grid[r][60] = 1
        for c in range(30, 36): self.grid[30][c] = 1
        for c in range(45, 65): self.grid[36][c] = 1
        for c in range(45, 53): self.grid[40][c] = 1
        for r in range(38, 48): self.grid[r][68] = 1
        for c in range(45, 53): self.grid[47][c] = 1

    def ejecutar_algoritmo(self):
        t0 = time.perf_counter()
        if self.algoritmo_id == 1:
            self.path, self.explored = run_bfs(self.start, self.goal, self.grid)
        elif self.algoritmo_id == 2:
            self.path, self.explored = run_dfs(self.start, self.goal, self.grid)
        elif self.algoritmo_id == 3:
            self.path, self.explored = run_a_star(self.start, self.goal, self.grid)
        elif self.algoritmo_id == 4:
            self.path, self.explored = run_dijkstra(self.start, self.goal, self.grid)
        elif self.algoritmo_id == 5:
            self.path, self.explored = run_greedy(self.start, self.goal, self.grid)

        self.tiempo_ms = (time.perf_counter() - t0) * 1000.0

        nombres_cortos = ["BFS", "DFS", "A*", "Dijkstra", "Greedy"]
        nombre = nombres_cortos[self.algoritmo_id - 1]
        self.historial.append({
            "alg": nombre,
            "path_len": len(self.path),
            "expl": len(self.explored),
            "time": self.tiempo_ms
        })
        if len(self.historial) > 4:
            self.historial.pop(0)

    def dibujar_grid(self):
        componente = set()
        if self.mostrar_componentes:
            componente = get_connected_component(self.start, self.grid)

        for r in range(ROWS):
            for c in range(COLS):
                rect = pygame.Rect(PANEL_WIDTH + c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)

                if self.grid[r][c] == 1:
                    color = COLOR_WALL
                elif (r, c) == self.start:
                    color = COLOR_START
                elif (r, c) == self.goal:
                    color = COLOR_GOAL
                elif (r, c) in self.path:
                    color = COLOR_PATH
                elif self.mostrar_componentes and (r, c) in componente:
                    color = COLOR_EXPLORED
                else:
                    color = COLOR_GRID_BG

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID_LINE, rect, 1)

        for npc in self.npcs:
            nr, nc = npc
            center = (PANEL_WIDTH + nc * CELL_SIZE + CELL_SIZE // 2, nr * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, COLOR_NPC, center, CELL_SIZE // 2 - 2)

    def dibujar_panel(self):
        panel_rect = pygame.Rect(0, 0, PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_BG_PANEL, panel_rect)
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (PANEL_WIDTH - 1, 0), (PANEL_WIDTH - 1, SCREEN_HEIGHT), 2)

        x = 16
        y = 12

        lbl_title = self.font_title.render("SISTEMA DE NAVEGACIÓN", True, TEXT_TITLE)
        self.screen.blit(lbl_title, (x, y))
        y += 24

        lbl_sub = self.font_subtitle.render("Pathfinding GUI (Dark Mode)", True, TEXT_SUB)
        self.screen.blit(lbl_sub, (x, y))
        y += 32

        # CONTROLES
        lbl_ctrl = self.font_header.render("CONTROLES", True, TEXT_HEADER)
        self.screen.blit(lbl_ctrl, (x, y))
        y += 20

        controles = [
            "1: BFS", "2: DFS", "3: A*", "4: Dijkstra", "5: Greedy", "",
            "C: Mostrar/Ocultar zonas alcanzables",
            "N: Crear NPC aleatorio",
            "M: Mover NPCs hacia la meta",
            "R: Reiniciar mapa", "",
            "Click Izq: Dibujar pared",
            "Click Der: Borrar pared",
            "ESC: Salir"
        ]
        for c in controles:
            if c == "":
                y += 6
                continue
            lbl = self.font_body.render(c, True, TEXT_BODY)
            self.screen.blit(lbl, (x, y))
            y += 16
        y += 10

        # ALGORITMO ACTIVO
        lbl_act = self.font_header.render("ALGORITMO ACTIVO", True, TEXT_HEADER)
        self.screen.blit(lbl_act, (x, y))
        y += 20

        nombre_activo = self.nombres_algoritmos.get(self.algoritmo_id, "")
        lbl_nom = self.font_body.render(nombre_activo, True, TEXT_HIGHLIGHT)
        self.screen.blit(lbl_nom, (x, y))
        y += 28

        # ESTADÍSTICAS
        lbl_est = self.font_header.render("ESTADÍSTICAS", True, TEXT_HEADER)
        self.screen.blit(lbl_est, (x, y))
        y += 20

        stats = [
            f"Ruta encontrada: {'Sí' if len(self.path) > 0 else 'No'}",
            f"Longitud: {len(self.path)} celdas",
            f"Nodos explorados: {len(self.explored)}",
            f"Tiempo: {self.tiempo_ms:.4f} ms"
        ]
        for s in stats:
            lbl = self.font_body.render(s, True, TEXT_BODY)
            self.screen.blit(lbl, (x, y))
            y += 17
        y += 10

        # INFORMACIÓN DEL MAPA
        lbl_info = self.font_header.render("INFORMACIÓN DEL MAPA", True, TEXT_HEADER)
        self.screen.blit(lbl_info, (x, y))
        y += 20

        total_celdas = ROWS * COLS
        obstaculos = sum(row.count(1) for row in self.grid)
        transitables = total_celdas - obstaculos

        map_info = [
            f"Tamaño: {COLS} x {ROWS}",
            f"Celdas totales: {total_celdas}",
            f"Nodos transitables: {transitables}",
            f"Obstáculos: {obstaculos}",
            f"NPCs activos: {len(self.npcs)}"
        ]
        for mi in map_info:
            lbl = self.font_body.render(mi, True, TEXT_BODY)
            self.screen.blit(lbl, (x, y))
            y += 17

        lbl_ini = self.font_body.render(f"Inicio: ({self.start[1]}, {self.start[0]})", True, COLOR_START)
        self.screen.blit(lbl_ini, (x, y))
        y += 17

        lbl_fin = self.font_body.render(f"Meta: ({self.goal[1]}, {self.goal[0]})", True, COLOR_GOAL)
        self.screen.blit(lbl_fin, (x, y))
        y += 22

        # COMPARATIVA
        lbl_comp = self.font_header.render("COMPARATIVA", True, TEXT_HEADER)
        self.screen.blit(lbl_comp, (x, y))
        y += 20

        if self.historial:
            header = f"{'Alg':<9}{'Long':<6}{'Expl':<6}{'T(ms)':<6}"
            lbl_head = self.font_mono.render(header, True, TEXT_TITLE)
            self.screen.blit(lbl_head, (x, y))
            y += 15
            for h in self.historial[-3:]:
                line = f"{h['alg']:<9}{h['path_len']:<6}{h['expl']:<6}{h['time']:<6.2f}"
                lbl_line = self.font_mono.render(line, True, TEXT_SUB)
                self.screen.blit(lbl_line, (x, y))
                y += 14

    def spawn_npc(self):
        componente = list(get_connected_component(self.start, self.grid))
        if componente:
            nueva_pos = random.choice(componente)
            if nueva_pos != self.start and nueva_pos != self.goal:
                self.npcs.append(nueva_pos)

    def mover_npcs(self):
        nuevas_posiciones = []
        for npc in self.npcs:
            if npc == self.goal:
                nuevas_posiciones.append(npc)
                continue
            camino, _ = run_a_star(npc, self.goal, self.grid)
            if camino and len(camino) > 0:
                nuevas_posiciones.append(camino[0])
            else:
                nuevas_posiciones.append(npc)
        self.npcs = nuevas_posiciones

    def run(self):
        running = True
        drawing = False
        erasing = False

        while running:
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_1:
                        self.algoritmo_id = 1
                        self.ejecutar_algoritmo()
                    elif event.key == pygame.K_2:
                        self.algoritmo_id = 2
                        self.ejecutar_algoritmo()
                    elif event.key == pygame.K_3:
                        self.algoritmo_id = 3
                        self.ejecutar_algoritmo()
                    elif event.key == pygame.K_4:
                        self.algoritmo_id = 4
                        self.ejecutar_algoritmo()
                    elif event.key == pygame.K_5:
                        self.algoritmo_id = 5
                        self.ejecutar_algoritmo()
                    elif event.key == pygame.K_c:
                        self.mostrar_componentes = not self.mostrar_componentes
                        self.ejecutar_algoritmo()
                    elif event.key == pygame.K_n:
                        self.spawn_npc()
                    elif event.key == pygame.K_m:
                        self.mover_npcs()
                    elif event.key == pygame.K_r:
                        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
                        self.npcs.clear()
                        self.ejecutar_algoritmo()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.pos[0] > PANEL_WIDTH:
                        if event.button == 1:
                            drawing = True
                        elif event.button == 3:
                            erasing = True

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        drawing = False
                    elif event.button == 3:
                        erasing = False

            if drawing or erasing:
                mx, my = pygame.mouse.get_pos()
                if mx > PANEL_WIDTH:
                    c = (mx - PANEL_WIDTH) // CELL_SIZE
                    r = my // CELL_SIZE
                    if 0 <= r < ROWS and 0 <= c < COLS:
                        if (r, c) != self.start and (r, c) != self.goal:
                            nuevo_valor = 1 if drawing else 0
                            if self.grid[r][c] != nuevo_valor:
                                self.grid[r][c] = nuevo_valor
                                self.ejecutar_algoritmo()

            self.screen.fill(COLOR_BG_PANEL)
            self.dibujar_grid()
            self.dibujar_panel()
            pygame.display.flip()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = PathfindingVisualizer()
    app.run()