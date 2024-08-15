from collections import namedtuple
import copy
import torch
import numpy as np


Move = namedtuple('Move', 'piece, cell, special')
Cell = namedtuple('Cell', 'row, col')


class CellUtils:
    @staticmethod
    def cell(cell):
        if type(cell) == Cell:
            return cell
        else:
            return Cell(8 - int(cell[1]), ord(cell[0]) - 97)

    @staticmethod
    def cell_code(cell):
        if type(cell) == Cell:
            return f'{chr(cell.col + 97)}{str(8 - cell.row)}'
        else:
            return cell


class Piece:
    def __init__(self, color, cell, notation, value):
        self.color = color
        self.row, self.col = cell.row, cell.col
        self.notation = notation.upper() if self.color == 0 else notation.lower()
        self.value = value
        self.attacks = []
        self.legal = []
        self.mobility = 0
        self.alive = 1

    def _is_pinned(self, board, move_to):
        board2 = board.copy()
        self_copy = board2.cells[self.row][self.col]
        killed = board2[move_to]
        board2.cells[self.row][self.col] = None
        board2[move_to] = self_copy
        if killed is not None:
            board2.material[1 - self.color].remove(killed)

        king_cell = Cell(board2.pieces[self.color]['k'][0].row, board2.pieces[self.color]['k'][0].col)
        for piece in board2.material[1-self.color]:
            piece.update_attacks(board2)
            if king_cell in piece.attacks:
                return True
        return False

    def update_attacks(self, board):
        pass

    def update_legals(self, board):
        self.legal = []
        for attacked_cell in self.attacks:
            move = Move(self, attacked_cell, None)
            self.legal.append(move)
            if board[attacked_cell] in board.material[self.color]:
                self.legal.remove(move)
            elif self._is_pinned(board, attacked_cell):
                self.legal.remove(move)
        self.mobility = len(self.legal)

    def get_slot(self, board, side):
        if side == 0:
            return [self.alive, self.row - 3.5, self.col - 3.5, self.mobility,
                    board.lowest_attackers[1 - self.color][self.row][self.col],
                    board.lowest_attackers[self.color][self.row][self.col],
                    ]
        else:
            return [self.alive, 3.5 - self.row, 3.5 - self.col, self.mobility,
                    board.lowest_attackers[1 - self.color][self.row][self.col],
                    board.lowest_attackers[self.color][self.row][self.col],
                    ]


class King(Piece):
    def __init__(self, color, cell):
        Piece.__init__(self, color, cell, 'K', 25)

    def update_attacks(self, board):
        self.attacks = []
        for verti in range(-1, 2):
            for horiz in range(-1, 2):
                if verti == 0 and horiz == 0:
                    pass
                elif self.row + verti > 7 or self.row + verti < 0 or self.col + horiz > 7 or self.col + horiz < 0:
                    pass
                else:
                    self.attacks.append(Cell(self.row + verti, self.col + horiz))

    def _castling_moves(self, board):
        moves = []
        if board.castling[self.color][0] == 1:
            if board.cells[self.row][5] is None and board.cells[self.row][6] is None and board.cells[self.row][4] is None:
                move = Move(self, Cell(self.row, 6), 'c')
                moves.append(move)
                for enemy in board.material[1-self.color]:
                    enemy_attacks = enemy.attacks
                    if Cell(self.row, 5) in enemy_attacks or Cell(self.row, 6) in enemy_attacks:
                        moves.remove(move)
                        break
        if board.castling[self.color][1] == 1:
            if board.cells[self.row][1] is None and board.cells[self.row][2] is None and board.cells[self.row][3] is None and board.cells[self.row][4] is None:
                move = Move(self, Cell(self.row, 2), 'c')
                moves.append(move)
                for enemy in board.material[1-self.color]:
                    enemy_attacks = enemy.attacks
                    if Cell(self.row, 1) in enemy_attacks or Cell(self.row, 2) in enemy_attacks or Cell(self.row, 3) in enemy_attacks:
                        moves.remove(move)
                        break
        return moves

    def update_legals(self, board):
        self.legal = []
        for attacked_cell in self.attacks:
            move = Move(self, attacked_cell, None)
            self.legal.append(move)
            if board[attacked_cell] in board.material[self.color]:
                self.legal.remove(move)
            else:
                board2 = board.copy()
                board2.cells[self.row][self.col] = None
                for piece in board2.material[1 - self.color]:
                    if attacked_cell in piece.attacks:
                        self.legal.remove(move)
                        break
                    elif Cell(self.row, self.col) in piece.attacks:
                        piece.update_attacks(board2)
                        if attacked_cell in piece.attacks:
                            self.legal.remove(move)
                            break
        self.legal += self._castling_moves(board)
        self.mobility = len(self.legal)

    def get_slot(self, board, side):
        if side == 0:
            return [self.row - 3.5, self.col - 3.5, self.mobility,
                    board.lowest_attackers[1 - self.color][self.row][self.col],
                    board.lowest_attackers[self.color][self.row][self.col],
                    ]
        else:
            return [3.5 - self.row, 3.5 - self.col, self.mobility,
                    board.lowest_attackers[1 - self.color][self.row][self.col],
                    board.lowest_attackers[self.color][self.row][self.col],
                    ]


class Queen(Piece):
    def __init__(self, color, cell):
        Piece.__init__(self, color, cell, 'Q', 9)

    def update_attacks(self, board):
        self.attacks = []
        cells = board.cells
        material = board.material
        for U in range(1, self.row + 1):
            if cells[self.row - U][self.col] is None:
                self.attacks.append(Cell(self.row - U, self.col))
            elif cells[self.row - U][self.col] in material[self.color]:
                self.attacks.append(Cell(self.row - U, self.col))
                break
            else:
                self.attacks.append(Cell(self.row - U, self.col))
                break
        for D in range(1, 8 - self.row):
            if cells[self.row + D][self.col] is None:
                self.attacks.append(Cell(self.row + D, self.col))
            elif cells[self.row + D][self.col] in material[self.color]:
                self.attacks.append(Cell(self.row + D, self.col))
                break
            else:
                self.attacks.append(Cell(self.row + D, self.col))
                break
        for R in range(1, 8 - self.col):
            if cells[self.row][self.col + R] is None:
                self.attacks.append(Cell(self.row, self.col + R))
            elif cells[self.row][self.col + R] in material[self.color]:
                self.attacks.append(Cell(self.row, self.col + R))
                break
            else:
                self.attacks.append(Cell(self.row, self.col + R))
                break
        for L in range(1, self.col + 1):
            if cells[self.row][self.col - L] is None:
                self.attacks.append(Cell(self.row, self.col - L))
            elif cells[self.row][self.col - L] in material[self.color]:
                self.attacks.append(Cell(self.row, self.col - L))
                break
            else:
                self.attacks.append(Cell(self.row, self.col - L))
                break
        for U_R in range(1, min(self.row + 1, 8 - self.col)):
            if cells[self.row - U_R][self.col + U_R] is None:
                self.attacks.append(Cell(self.row - U_R, self.col + U_R))
            elif cells[self.row - U_R][self.col + U_R] in material[self.color]:
                self.attacks.append(Cell(self.row - U_R, self.col + U_R))
                break
            else:
                self.attacks.append(Cell(self.row - U_R, self.col + U_R))
                break
        for U_L in range(1, min(self.row + 1, self.col + 1)):
            if cells[self.row - U_L][self.col - U_L] is None:
                self.attacks.append(Cell(self.row - U_L, self.col - U_L))
            elif cells[self.row - U_L][self.col - U_L] in material[self.color]:
                self.attacks.append(Cell(self.row - U_L, self.col - U_L))
                break
            else:
                self.attacks.append(Cell(self.row - U_L, self.col - U_L))
                break
        for D_R in range(1, min(8 - self.row, 8 - self.col)):
            if cells[self.row + D_R][self.col + D_R] is None:
                self.attacks.append(Cell(self.row + D_R, self.col + D_R))
            elif cells[self.row + D_R][self.col + D_R] in material[self.color]:
                self.attacks.append(Cell(self.row + D_R, self.col + D_R))
                break
            else:
                self.attacks.append(Cell(self.row + D_R, self.col + D_R))
                break
        for D_L in range(1, min(8 - self.row, self.col + 1)):
            if cells[self.row + D_L][self.col - D_L] is None:
                self.attacks.append(Cell(self.row + D_L, self.col - D_L))
            elif cells[self.row + D_L][self.col - D_L] in material[self.color]:
                self.attacks.append(Cell(self.row + D_L, self.col - D_L))
                break
            else:
                self.attacks.append(Cell(self.row + D_L, self.col - D_L))
                break


class Rook(Piece):
    def __init__(self, color, cell):
        Piece.__init__(self, color, cell, 'R', 5)

    def update_attacks(self, board):
        self.attacks = []
        cells = board.cells
        material = board.material
        for U in range(1, self.row + 1):
            if cells[self.row - U][self.col] is None:
                self.attacks.append(Cell(self.row - U, self.col))
            elif cells[self.row - U][self.col] in material[self.color]:
                self.attacks.append(Cell(self.row - U, self.col))
                break
            else:
                self.attacks.append(Cell(self.row - U, self.col))
                break
        for D in range(1, 8 - self.row):
            if cells[self.row + D][self.col] is None:
                self.attacks.append(Cell(self.row + D, self.col))
            elif cells[self.row + D][self.col] in material[self.color]:
                self.attacks.append(Cell(self.row + D, self.col))
                break
            else:
                self.attacks.append(Cell(self.row + D, self.col))
                break
        for R in range(1, 8 - self.col):
            if cells[self.row][self.col + R] is None:
                self.attacks.append(Cell(self.row, self.col + R))
            elif cells[self.row][self.col + R] in material[self.color]:
                self.attacks.append(Cell(self.row, self.col + R))
                break
            else:
                self.attacks.append(Cell(self.row, self.col + R))
                break
        for L in range(1, self.col + 1):
            if cells[self.row][self.col - L] is None:
                self.attacks.append(Cell(self.row, self.col - L))
            elif cells[self.row][self.col - L] in material[self.color]:
                self.attacks.append(Cell(self.row, self.col - L))
                break
            else:
                self.attacks.append(Cell(self.row, self.col - L))
                break


class Bishop(Piece):
    def __init__(self, color, cell):
        Piece.__init__(self, color, cell, 'B', 3)

    def update_attacks(self, board):
        self.attacks = []
        cells = board.cells
        material = board.material
        for U_R in range(1, min(self.row + 1, 8 - self.col)):
            if cells[self.row - U_R][self.col + U_R] is None:
                self.attacks.append(Cell(self.row - U_R, self.col + U_R))
            elif cells[self.row - U_R][self.col + U_R] in material[self.color]:
                self.attacks.append(Cell(self.row - U_R, self.col + U_R))
                break
            else:
                self.attacks.append(Cell(self.row - U_R, self.col + U_R))
                break
        for U_L in range(1, min(self.row + 1, self.col + 1)):
            if cells[self.row - U_L][self.col - U_L] is None:
                self.attacks.append(Cell(self.row - U_L, self.col - U_L))
            elif cells[self.row - U_L][self.col - U_L] in material[self.color]:
                self.attacks.append(Cell(self.row - U_L, self.col - U_L))
                break
            else:
                self.attacks.append(Cell(self.row - U_L, self.col - U_L))
                break
        for D_R in range(1, min(8 - self.row, 8 - self.col)):
            if cells[self.row + D_R][self.col + D_R] is None:
                self.attacks.append(Cell(self.row + D_R, self.col + D_R))
            elif cells[self.row + D_R][self.col + D_R] in material[self.color]:
                self.attacks.append(Cell(self.row + D_R, self.col + D_R))
                break
            else:
                self.attacks.append(Cell(self.row + D_R, self.col + D_R))
                break
        for D_L in range(1, min(8 - self.row, self.col + 1)):
            if cells[self.row + D_L][self.col - D_L] is None:
                self.attacks.append(Cell(self.row + D_L, self.col - D_L))
            elif cells[self.row + D_L][self.col - D_L] in material[self.color]:
                self.attacks.append(Cell(self.row + D_L, self.col - D_L))
                break
            else:
                self.attacks.append(Cell(self.row + D_L, self.col - D_L))
                break


class Knight(Piece):
    def __init__(self, color, cell):
        Piece.__init__(self, color, cell, 'N', 3)

    def update_attacks(self, board):
        self.attacks = []
        for move1 in [-2, 2]:
            for move2 in [-1, 1]:
                if self.row + move1 < 0 or self.row + move1 > 7 or self.col + move2 < 0 or self.col + move2 > 7:
                    pass
                else:
                    self.attacks.append(Cell(self.row + move1, self.col + move2))
                if self.row + move2 < 0 or self.row + move2 > 7 or self.col + move1 < 0 or self.col + move1 > 7:
                    pass
                else:
                    self.attacks.append(Cell(self.row + move2, self.col + move1))


class Pawn(Piece):
    def __init__(self, color, cell):
        Piece.__init__(self, color, cell, 'P', 1)

    def update_attacks(self, board):
        self.attacks = []
        if self.col - 1 >= 0:
            self.attacks.append(Cell(self.row + 2*self.color - 1, self.col - 1))
        if self.col + 1 <= 7:
            self.attacks.append(Cell(self.row + 2*self.color - 1, self.col + 1))

    def update_legals(self, board):
        self.legal = []

        for attack in self.attacks:
            row, col = attack.row, attack.col
            if board.cells[row][col] in board.material[1 - self.color] or board.enpassant == attack:
                self.legal.append(Move(self, attack, None))

        if board.cells[self.row + 2*self.color - 1][self.col] is None:
            self.legal.append(Move(self, Cell(self.row + 2*self.color - 1, self.col), None))
            if self.row == 6 - 5 * self.color and board.cells[self.row + 2 * (2*self.color - 1)][self.col] is None:
                self.legal.append(Move(self, Cell(self.row + 2 * (2*self.color - 1), self.col), None))

        temp = self.legal.copy()
        for move in temp:
            attacked_cell = move.cell
            if board[attacked_cell] in board.material[self.color]:
                self.legal.remove(move)
            elif self._is_pinned(board, attacked_cell):
                self.legal.remove(move)

        if self.row == 1 + 5 * self.color:
            temp = []
            for move in self.legal:
                for promote_to in ['q', 'b', 'r', 'n']:
                    temp.append(Move(self, move.cell, promote_to))
            self.legal = temp
        self.mobility = len(self.legal)


class Board:
    def __init__(self, fen: str = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
        self.pieces = {
            0: {'k': [], 'q': [], 'r': [], 'n': [], 'b': [], 'p': []},
            1: {'k': [], 'q': [], 'r': [], 'n': [], 'b': [], 'p': []},
        }
        self.pieces_for_piece_list = {
            0: {'k': [], 'q': [], 'r': [], 'n': [], 'b': [], 'p': []},
            1: {'k': [], 'q': [], 'r': [], 'n': [], 'b': [], 'p': []},
        }
        self.material = {
            0: [],
            1: []
        }
        self.cells = []
        self.side_to_move = None
        self.castling = [[0, 0], [0, 0]]    # (short, long) castle
        self.enpassant = None
        self.lowest_attackers = None
        piece_position, side, castling, enpassant_target, half_moves, full_moves = fen.split(' ')
        self._read_cells(piece_position)
        self._read_side_to_play(side)
        self._read_castling(castling)
        self._read_enpassant(enpassant_target)
        self.half_moves = int(half_moves)
        self._update_moves()
        self._update_attack_maps()

    def _read_cells(self, piece_position):
        rows = piece_position.split('/')
        for row in rows:
            r = len(self.cells)
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend([None] * int(char))
                else:
                    c = len(board_row)
                    piece = None
                    if char.isupper():
                        color = 0
                    else:
                        color = 1
                    match char.lower():
                        case 'k':
                            piece = King(color, Cell(r, c))
                            self.pieces[color]['k'].append(piece)
                            self.pieces_for_piece_list[color]['k'].append(piece)
                        case 'q':
                            piece = Queen(color, Cell(r, c))
                            self.pieces[color]['q'].append(piece)
                            self.pieces_for_piece_list[color]['q'].append(piece)
                        case 'r':
                            piece = Rook(color, Cell(r, c))
                            self.pieces[color]['r'].append(piece)
                            self.pieces_for_piece_list[color]['r'].append(piece)
                        case 'n':
                            piece = Knight(color, Cell(r, c))
                            self.pieces[color]['n'].append(piece)
                            self.pieces_for_piece_list[color]['n'].append(piece)
                        case 'b':
                            piece = Bishop(color, Cell(r, c))
                            self.pieces[color]['b'].append(piece)
                            self.pieces_for_piece_list[color]['b'].append(piece)
                        case 'p':
                            piece = Pawn(color, Cell(r, c))
                            self.pieces[color]['p'].append(piece)
                            self.pieces_for_piece_list[color]['p'].append(piece)
                    self.material[color].append(piece)
                    board_row.append(piece)
            self.cells.append(board_row)

    def _read_side_to_play(self, side):
        if side == 'w':
            self.side_to_move = 0
        elif side == 'b':
            self.side_to_move = 1

    def _read_castling(self, castling):
        if 'K' in castling:
            self.castling[0][0] = 1
        if 'Q' in castling:
            self.castling[0][1] = 1
        if 'k' in castling:
            self.castling[1][0] = 1
        if 'q' in castling:
            self.castling[1][1] = 1

    def _read_enpassant(self, enpassant_target):
        if enpassant_target != '-':
            self.enpassant = CellUtils.cell(enpassant_target)

    def _update_moves(self):
        for color in range(2):
            for piece_type in ['q', 'r', 'n', 'b', 'p']:
                for piece in self.pieces[color][piece_type]:
                    piece.update_attacks(self)
        for color in range(2):
            for piece_type in ['q', 'r', 'n', 'b', 'p']:
                for piece in self.pieces[color][piece_type]:
                    piece.update_legals(self)
        for color in range(2):
            self.pieces[color]['k'][0].update_attacks(self)
        for color in range(2):
            self.pieces[color]['k'][0].update_legals(self)
            # for piece_type, pieces in self.pieces[color].items():
            #     for piece in pieces:
            #         piece.update_attacks(self)
        # for color in range(2):
        #     for piece_type, pieces in self.pieces[color].items():
        #         for piece in pieces:
        #             piece.update_legals(self)

    def _update_attack_maps(self):
        self.lowest_attackers = {
            0: [
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50]
            ],
            1: [
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [50, 50, 50, 50, 50, 50, 50, 50]
            ]
        }
        for color in range(2):
            for piece in self.material[color]:
                for attack in piece.attacks:
                    self.lowest_attackers[color][attack.row][attack.col] = min(
                        self.lowest_attackers[color][attack.row][attack.col], piece.value
                    )

    def __getitem__(self, cell):
        cell = CellUtils.cell(cell)
        return self.cells[cell.row][cell.col]

    def __setitem__(self, cell, piece):
        cell = CellUtils.cell(cell)
        self.cells[cell.row][cell.col] = piece

    def copy(self):
        return copy.deepcopy(self)

    def display(self):
        print(" " * 4, end="")
        print("+-----" * 8 + "+")
        for row_num, row in enumerate(self.cells):
            print(f" {8 - row_num}  ", end="")
            for item in row:
                if item is None:
                    notation = " "
                else:
                    notation = item.notation
                print(f"|  {notation}  ", end="")
            print("|")
            print(" " * 4, end="")
            print("+-----" * 8 + "+")
        print(" " * 4, end="")
        print("   a     b     c     d     e     f     g     h")

    def _promote(self, move):
        self._kill_piece(self.cells[move.piece.row][move.piece.col])
        self.cells[move.piece.row][move.piece.col] = None
        color = move.piece.color
        r, c = move.cell.row, move.cell.col
        piece = None
        match move.special:
            case 'q':
                piece = Queen(color, Cell(r, c))
                self.pieces[color]['q'].append(piece)
            case 'r':
                piece = Rook(color, Cell(r, c))
                self.pieces[color]['r'].append(piece)
            case 'n':
                piece = Knight(color, Cell(r, c))
                self.pieces[color]['n'].append(piece)
            case 'b':
                piece = Bishop(color, Cell(r, c))
                self.pieces[color]['b'].append(piece)
        self.material[color].append(piece)

        if self.cells[r][c] is not None:
            self._kill_piece(self.cells[r][c])

        for temp in self.pieces_for_piece_list[color][move.special]:
            if temp.alive == 0:
                self.pieces_for_piece_list[color][move.special].remove(temp)
                self.pieces_for_piece_list[color][move.special].append(piece)
                break

        self.cells[r][c] = piece
        piece.update_attacks(self)
        piece.update_legals(self)

    def _update_castling(self, move):
        piece = move.piece
        if type(piece) == King:
            self.castling[piece.color][0] = 0
            self.castling[piece.color][1] = 0
        elif type(piece) == Rook:
            if piece.col == 7:
                self.castling[piece.color][0] = 0
            elif piece.col == 0:
                self.castling[piece.color][1] = 0

    def _castle(self, move):
        col = move.cell.col
        king_move = Move(move.piece, move.cell, None)
        self.apply_move(king_move, True)
        if col == 6:
            for rook in self.pieces[move.piece.color]['r']:
                if rook.col == 7:
                    rook_move = Move(rook, Cell(move.cell.row, 5), None)
                    self.apply_move(rook_move, True)
                    break
        else:
            for rook in self.pieces[move.piece.color]['r']:
                if rook.col == 0:
                    rook_move = Move(rook, Cell(move.cell.row, 3), None)
                    self.apply_move(rook_move, True)
                    break
        self._update_moves()

    def _apply_special_move(self, move):
        if move.special == 'c':
            self._castle(move)
        else:
            self._promote(move)
            self._update_moves()

    def _kill_piece(self, killed):
        killed.alive = 0
        self.material[killed.color].remove(killed)
        self.pieces[killed.color][killed.notation.lower()].remove(killed)

    def apply_move(self, move, inplace=False):
        board = self if inplace else self.copy()
        was_check = False
        reward = 0
        done = False
        if board.is_check(move.piece.color):
            was_check = True

        if move.special is None:
            start_row, start_col = move.piece.row, move.piece.col
            end_row, end_col = move.cell.row, move.cell.col
            if board.cells[end_row][end_col] is not None:
                board._kill_piece(board.cells[end_row][end_col])
                board.half_moves = -1
            board.cells[end_row][end_col] = board.cells[start_row][start_col]
            board.cells[start_row][start_col] = None
            board.cells[end_row][end_col].row, board.cells[end_row][end_col].col = end_row, end_col
            board._update_moves()
            if type(move.piece) == Pawn:
                board.half_moves = -1

        else:
            board.half_moves = -1
            board._apply_special_move(move)

        board.side_to_move = 1 - board.side_to_move
        board._update_castling(move)
        board._update_attack_maps()
        board.half_moves += 1
        if board.is_check(board.side_to_move):
            board._update_moves()
            if board.is_checkmate(board.side_to_move):
                done = True
                reward = 1
        elif board.is_draw():
            done = True
        if was_check:
            board._update_moves()
        return board, reward, done

    def legal_moves(self, color):
        moves = []
        for piece in self.material[color]:
            moves += piece.legal
        return moves

    def is_check(self, color):
        king_cell = Cell(self.pieces[color]['k'][0].row, self.pieces[color]['k'][0].col)
        for piece in self.material[1 - color]:
            if king_cell in piece.attacks:
                return True
        return False

    def is_checkmate(self, color):
        checkmate = False
        if self.is_check(color):
            checkmate = True
            for piece in self.material[color]:
                if len(piece.legal) > 0:
                    checkmate = False
        return checkmate

    def is_stalemate(self, color):
        for soldier in self.material[color]:
            if len(soldier.legal) > 0:
                return False
        return True

    def is_fifty_move_draw(self):
        if self.half_moves >= 50:
            return True
        return False

    def is_draw(self):
        return self.is_fifty_move_draw() or self.is_stalemate(self.side_to_move)

    def _get_piece_features(self, side):
        result = []
        for color in [side, 1-side]:
            result += self.pieces_for_piece_list[color]['k'][0].get_slot(self, side)
            if len(self.pieces_for_piece_list[color]['q']) == 0:
                result += [0, 0, 0, 0, 50, 50]
            else:
                result += self.pieces_for_piece_list[color]['q'][0].get_slot(self, side)

            for i in [side, 1-side]:
                if i >= len(self.pieces_for_piece_list[color]['r']):
                    result += [0, 0, 0, 0, 50, 50]
                else:
                    result += self.pieces_for_piece_list[color]['r'][i].get_slot(self, side)

            for i in [side, 1-side]:
                if i >= len(self.pieces_for_piece_list[color]['n']):
                    result += [0, 0, 0, 0, 50, 50]
                else:
                    result += self.pieces_for_piece_list[color]['n'][i].get_slot(self, side)

            for i in [side, 1-side]:
                if i >= len(self.pieces_for_piece_list[color]['b']):
                    result += [0, 0, 0, 0, 50, 50]
                else:
                    result += self.pieces_for_piece_list[color]['b'][i].get_slot(self, side)

            if side == 0:
                for i in range(8):
                    if i >= len(self.pieces_for_piece_list[color]['p']):
                        result += [0, 0, 0, 0, 50, 50]
                    else:
                        result += self.pieces_for_piece_list[color]['p'][i].get_slot(self, side)
            else:
                for i in range(7, -1, -1):
                    if i >= len(self.pieces_for_piece_list[color]['p']):
                        result += [0, 0, 0, 0, 50, 50]
                    else:
                        result += self.pieces_for_piece_list[color]['p'][i].get_slot(self, side)
        return result

    def _get_global_features(self, side):
        result = [abs(self.side_to_move - side)]
        for color in [side, 1-side]:
            for c_side in range(2):
                result.append(self.castling[color][c_side])

        for color in [side, 1-side]:
            for piece in ['q', 'r', 'n', 'b', 'p']:
                result.append(len(self.pieces[color][piece]))
        return result

    def _get_attack_maps(self, side):
        result = []
        if side == 0:
            for color in range(2):
                for row in range(8):
                    for col in range(8):
                        result.append(self.lowest_attackers[1-color][row][col])
        else:
            for color in range(2):
                for row in range(7, -1, -1):
                    for col in range(7, -1, -1):
                        result.append(self.lowest_attackers[color][row][col])
        return result

    def get_state(self, side):
        piece_features = self._get_piece_features(side)
        global_features = self._get_global_features(side)
        attack_map_features = self._get_attack_maps(side)
        return torch.from_numpy(np.array(global_features + piece_features + attack_map_features).astype(np.float32))


def play():
    board = Board()
    while True:
        board.display()
        side = board.side_to_move
        color = "WHITE" if side == 0 else "BLACK"
        print(color + " TO MOVE")
        start = input(">> start cell:\t")
        try:
            start = CellUtils.cell(start)
        except ValueError:
            print(">> invalid cell!")
            continue
        except IndexError:
            print(">> invalid cell!")
            continue

        moving = None
        for piece in board.material[side]:
            if piece.row == start.row and piece.col == start.col:
                moving = piece
                break
        if moving is None:
            print(">> invalid cell!")
            continue

        i = 1
        print(">> valid moves:")
        for legal in moving.legal:
            if legal.special is None:
                print(f'[{i}]. ' + CellUtils.cell_code(legal.cell))
            else:
                print(f'[{i}]. ' + CellUtils.cell_code(legal.cell) + f' ({legal.special})')
            i += 1
        end = input(">> select move:\t")
        try:
            move = board[start].legal[int(end) - 1]
        except ValueError:
            print(">> invalid move!")
            continue
        except IndexError:
            print(">> invalid move!")
            continue

        board.apply_move(move, True)

        if board.is_checkmate(1-side):
            print(">> CHECKMATE!")
            print(color + " WINS!!!")
            break
        elif board.is_check(1-side):
            print(">> CHECK!")
        elif board.is_draw():
            print(">> DRAW!")
            break


def testing():
    board = Board()
    board.display()
    board.get_state(0)


if __name__ == '__main__':
    # play()
    testing()
