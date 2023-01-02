import tkinter as tk
import colors as c
import main_func


class Game(tk.Frame):
    def __init__(self):
        tk.Frame.__init__(self)
        self.path = None
        self.path_label = None
        self.cells = None
        self.matrix = None
        self.position_row = 3
        self.position_col = 3
        self.grid()
        self.master.title('15-sliding-block puzzle')
        self.main_grid = tk.Frame(
            self, bg=c.GRID_COLOR, bd=3, width=400, height=400)
        self.main_grid.grid(pady=(80, 0))
        self.make_GUI()
        self.start_game()

        self.master.bind("<Left>", self.left)
        self.master.bind("<Right>", self.right)
        self.master.bind("<Up>", self.up)
        self.master.bind("<Down>", self.down)
        self.master.bind("<Return>", self.solve)

        self.mainloop()

    def make_GUI(self):
        # make grid
        self.cells = []
        for i in range(4):
            row = []
            for j in range(4):
                cell_frame = tk.Frame(
                    self.main_grid,
                    bg=c.EMPTY_CELL_COLOR,
                    width=100,
                    height=100)
                cell_frame.grid(row=i, column=j, padx=5, pady=5)
                cell_number = tk.Label(self.main_grid, bg=c.EMPTY_CELL_COLOR)
                cell_number.grid(row=i, column=j)
                cell_data = {"frame": cell_frame, "number": cell_number}
                row.append(cell_data)
            self.cells.append(row)

        # make score header
        path_frame = tk.Frame(self)
        path_frame.place(relx=0.5, y=40, anchor="center")
        tk.Label(
            path_frame,
            text="Path:",
            font=c.SCORE_LABEL_FONT).grid(
            row=0)
        self.path_label = tk.Label(path_frame, text="", font=c.SCORE_FONT)
        self.path_label.grid(row=1)

    def start_game(self):
        # create matrix of zeroes
        self.matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]

        for row in range(4):
            for col in range(4):
                val = self.matrix[row][col]
                self.cells[row][col]["frame"].configure(bg=c.CELL_COLORS)
                self.cells[row][col]["number"].configure(
                    bg=c.CELL_COLORS,
                    fg=c.CELL_NUMBER_COLORS,
                    font=c.CELL_NUMBER_FONTS,
                    text=str(val))

        self.path = ""

    def update_GUI(self):
        for row in range(4):
            for col in range(4):
                val = self.matrix[row][col]
                self.cells[row][col]["frame"].configure(bg=c.CELL_COLORS)
                self.cells[row][col]["number"].configure(
                    bg=c.CELL_COLORS,
                    fg=c.CELL_NUMBER_COLORS,
                    font=c.CELL_NUMBER_FONTS,
                    text=str(val))
        self.path_label.configure(text=self.path)
        self.update_idletasks()

    # Arrow-Press Functions

    def left(self, event):
        if self.position_col > 0:
            tmp = self.matrix[self.position_row][self.position_col]
            self.matrix[self.position_row][self.position_col] = self.matrix[self.position_row][self.position_col - 1]
            self.matrix[self.position_row][self.position_col - 1] = tmp
            self.position_col -= 1
            self.update_GUI()

    def right(self, event):
        if self.position_col < 3:
            tmp = self.matrix[self.position_row][self.position_col]
            self.matrix[self.position_row][self.position_col] = self.matrix[self.position_row][self.position_col + 1]
            self.matrix[self.position_row][self.position_col + 1] = tmp
            self.position_col += 1
            self.update_GUI()

    def up(self, event):
        if self.position_row > 0:
            tmp = self.matrix[self.position_row][self.position_col]
            self.matrix[self.position_row][self.position_col] = self.matrix[self.position_row - 1][self.position_col]
            self.matrix[self.position_row - 1][self.position_col] = tmp
            self.position_row -= 1
            self.update_GUI()

    def down(self, event):
        if self.position_row < 3:
            tmp = self.matrix[self.position_row][self.position_col]
            self.matrix[self.position_row][self.position_col] = self.matrix[self.position_row + 1][self.position_col]
            self.matrix[self.position_row + 1][self.position_col] = tmp
            self.position_row += 1
            self.update_GUI()

    def solve(self, event):
        self.path = main_func.a_star_search(main_func.Node(self.matrix), main_func.Node(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]), 3, 4000)
        print(self.path)
        self.update_GUI()


def main():
    Game()


if __name__ == "__main__":
    main()
