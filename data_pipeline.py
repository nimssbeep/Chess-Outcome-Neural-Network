import chess
import numpy as np

PIECE_VALUES = {
    chess.PAWN: 1, 
    chess.KNIGHT: 3, 
    chess.BISHOP: 3, 
    chess.ROOK: 5, 
    chess.QUEEN: 9, 
    chess.KING: 100
}

def board_to_vector(board):
    """Convert a python-chess Board object into a 64-element numeric vector."""
    vector = np.zeros(64)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = PIECE_VALUES[piece.piece_type]
            vector[square] = value if piece.color == chess.WHITE else -value
    return vector

def load_chess_dataset(num_samples=500):
    """Generate random chess-like board positions with clear material advantages.
    
    Returns:
        X: numpy array of shape (num_samples, 64) — board vectors
        y: numpy array of shape (num_samples, 1) — labels (1 = white wins, 0 = black wins)
    """
    rng = np.random.default_rng(42)
    X_data = np.zeros((num_samples, 64))
    y_data = np.zeros((num_samples, 1))

    for i in range(num_samples):
        board = np.zeros(64)
        # Place a random number of pieces on random squares
        num_pieces = rng.integers(4, 16)
        squares = rng.choice(64, size=num_pieces, replace=False)

        if i < num_samples // 2:
            # White is winning
            for sq in squares:
                board[sq] = rng.choice([1, 3, 5, 9]) * rng.choice([1, 1, 1, -1])
            board[squares[0]] = abs(board[squares[0]]) + 5  # ensure white advantage
            y_data[i] = 1
        else:
            # Black is winning
            for sq in squares:
                board[sq] = rng.choice([-1, -3, -5, -9]) * rng.choice([1, 1, 1, -1])
            board[squares[0]] = -(abs(board[squares[0]]) + 5)  # ensure black advantage
            y_data[i] = 0

        # Adding noise
        board += rng.normal(0, 0.1, 64)
        X_data[i] = board

    return X_data, y_data