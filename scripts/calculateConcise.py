import zlib
import sys

def compress(input):
	return zlib.compress(input.encode())

def main():
	# Example docstrings
	docstring1 = """
    Solves the N-Queens problem using backtracking.

    Args:
        n (int): The number of queens to place on an N x N chessboard.

    Returns:
        list of lists of tuples: Solutions, each representing queen positions.
    """
    
	docstring2 = """
    Solves the N-Queens problem recursively using backtracking.

    Args:
        n (int): The number of queens to place on an N x N chessboard.

    Returns:
        list of lists of tuples: A list of solutions, where each solution is
        represented as a list of (row, column) tuples specifying the positions
        of queens on the board.
    """
	
	comp1 = compress(docstring1)
	comp2 = compress(docstring2)
	
	print(sys.getsizeof(comp1) / sys.getsizeof(comp2))
	
if __name__ == "__main__":
	main()
