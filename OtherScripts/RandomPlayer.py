import chess
import random
import time

P_input_vec = []
R_input_vec = []
N_input_vec = []
B_input_vec = []
Q_input_vec = []
K_input_vec = []

p_input_vec = []
r_input_vec = []
n_input_vec = []
b_input_vec = []
q_input_vec = []
k_input_vec = []

class RandomPlayer(object):

	def __init__(self, board):

		self.board = board

	def make_board(self):

		return chess.Board()

	def choose_move(self):

		move = random.choice(list(self.board.legal_moves))

		return move

	def update_board(self,move):

		return self.board.push(move)

	def bitmapper(self):

		P_input_vec.append(self.board.pieces(chess.PAWN, chess.WHITE))
		print(self.board.pieces(chess.PAWN, chess.WHITE))
		R_input_vec.append(self.board.pieces(chess.ROOK, chess.WHITE))
		N_input_vec.append(self.board.pieces(chess.KNIGHT, chess.WHITE))
		B_input_vec.append(self.board.pieces(chess.BISHOP, chess.WHITE))
		Q_input_vec.append(self.board.pieces(chess.QUEEN, chess.WHITE))
		K_input_vec.append(self.board.pieces(chess.KING, chess.WHITE))

		p_input_vec.append(self.board.pieces(chess.PAWN, chess.BLACK))
		r_input_vec.append(self.board.pieces(chess.ROOK, chess.BLACK))
		n_input_vec.append(self.board.pieces(chess.KNIGHT, chess.BLACK))
		b_input_vec.append(self.board.pieces(chess.BISHOP, chess.BLACK))
		q_input_vec.append(self.board.pieces(chess.QUEEN, chess.BLACK))
	 	k_input_vec.append(self.board.pieces(chess.KING, chess.BLACK))

if __name__ == '__main__':

	p = RandomPlayer(chess.Board())

	while not p.board.is_game_over(True):
		p.update_board(p.choose_move())
		p.bitmapper()
		
		print(p.board)
		print "-----------------"
		time.sleep(0.5)

	print(p.board.result(True))		