import struct

# The struct format string:
# '<'  = Little-endian (standard for most CPUs, ensures compatibility with C++)
# 'Q'  = 8 bytes (unsigned long long) -> Occupancy
# '32s'= 32 bytes (char array)        -> Mailbox
# 'h'  = 2 bytes (signed short)       -> Packed Score + WDL
# 'B'  = 1 byte (unsigned char)       -> Friendly King
# 'B'  = 1 byte (unsigned char)       -> Enemy King
# '4x' = 4 bytes of padding           -> Padding to reach 48 bytes
struct_format = '<Q32shBB4x'

def parse_fen(fen):
    """
    Parses a FEN string into a 64-element array and STM flag.
    Returns:
        board_array: list of 64 ints (0-5 White, 6-11 Black, 15 Empty)
        stm_flag: 0 for White, 1 for Black
    """
    # We only care about the board and the side-to-move
    parts = fen.split()
    board_part = parts[0]
    stm_part = parts[1]
    
    # 0 = White, 1 = Black
    stm_flag = 0 if stm_part == 'w' else 1
    
    # Standard piece mapping. K=5 and k=11 to match our King-finding logic.
    piece_map = {
        'P': 0, 'B': 1, 'Q': 2, 'N': 3, 'R': 4, 'K': 5,
        'p': 6, 'b': 7, 'q': 8, 'n': 9, 'r': 10, 'k': 11
    }
    
    # Initialize the board with 15 (Empty)
    board_array = [15] * 64
    
    # FEN starts at a8 (Rank 7, File 0)
    rank = 7
    file = 0
    
    for char in board_part:
        if char == '/':
            # Move down a rank and reset to the a-file
            rank -= 1
            file = 0
        elif char.isdigit():
            # Skip empty squares
            file += int(char)
        else:
            # Calculate the 0-63 index (e.g., a1 = 0, a8 = 56)
            sq = rank * 8 + file
            board_array[sq] = piece_map[char]
            file += 1
            
    return board_array, stm_flag

def pack_board_state(board, stm):
    """
    Packs a 64-square board into normalized STM perspective.
    
    Returns:
        mailbox: 32 bytes (packed 4-bit nibbles)
        occupancy: int (64-bit integer bitboard)
        friendly_king: int (0-63 square index)
        enemy_king: int (0-63 square index)
    """
    mailbox = bytearray([0xFF] * 32)
    occupancy = 0
    friendly_king = 255 # 255 used as a safe "not found" default
    enemy_king = 255
    
    for sq in range(64):
        piece = board[sq]
        
        # Skip empty squares
        if piece is None or piece == 15:
            continue
            
        # 1. Normalize the Piece (STM = 0-5, NSTM = 6-11)
        if stm == 1:
            norm_piece = (piece + 6) if piece < 6 else (piece - 6)
        else:
            norm_piece = piece
            
        # 2. Normalize the Square (Mirror vertically if Black to move)
        norm_sq = (sq ^ 56) if stm == 1 else sq
        
        # 3. Pack the nibble into the correct byte
        byte_index = norm_sq // 2
        if norm_sq % 2 == 0:
            mailbox[byte_index] = (mailbox[byte_index] & 0xF0) | norm_piece
        else:
            mailbox[byte_index] = (mailbox[byte_index] & 0x0F) | (norm_piece << 4)
            
        # 4. Set the Occupancy Bit
        occupancy |= (1 << norm_sq)
        
        # 5. Extract King Squares (Assuming 5 = STM King, 11 = NSTM King)
        if norm_piece == 5:
            friendly_king = norm_sq
        elif norm_piece == 11:
            enemy_king = norm_sq
            
    return bytes(mailbox), occupancy, friendly_king, enemy_king

def convert_text_to_bin(input_txt, output_bin):
    with open(input_txt, 'r') as infile, open(output_bin, 'ab') as outfile:
        for line in infile:
            fen, score_str, result_str = line.strip().split('|')
            
            board, stm = parse_fen(fen)
            mailbox_bytes, occupancy, friendly_king_sq, enemy_king_sq = pack_board_state(board, stm)
            # 2. Prepare the numerical data
            score = int(score_str)
            # Assuming you map result to 0 (Loss), 1 (Draw), 2 (Win)
            wdl = 2 if result_str.strip() == "1.0" else 0 if result_str.strip() == "0.0" else 1 
            
            # 3. Flip score and WDL if it's Black's turn to move (to make it STM-relative)
            is_black_to_move = ' w ' not in fen
            if is_black_to_move:
                score = -score
                wdl = 2 - wdl # Flips Win to Loss, leaves Draw as Draw
                
            packed_score_wdl = (3 * score) + wdl
            
            # 4. Pack and write to disk
            # (Assuming `occupancy` is an int, and `mailbox_bytes` is a bytes object of len 32)
            binary_data = struct.pack(
                struct_format, 
                occupancy, 
                mailbox_bytes, 
                packed_score_wdl, 
                friendly_king_sq, 
                enemy_king_sq
            )
            outfile.write(binary_data)

for i in range(4):
    convert_text_to_bin(str(i)+".txt", "data.bin")