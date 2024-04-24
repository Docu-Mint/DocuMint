# pip install bert-score

from bert_score import score

# Calculate BERT encoding score, using cosine similarity
def calculate_bert_score(docstring1, docstring2):
    # Calculate BERT score
    _, _, bert_score_f1 = score([docstring1], [docstring2], lang='en', model_type='bert-base-uncased')

    return bert_score_f1.item()

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

    # Calculate BERT score
    bert_score = calculate_bert_score(docstring1, docstring2)
    print("BERT score:", bert_score)

if __name__ == "__main__":
    main()
