import chess.pgn

pgn = open("/home/matthia/Desktop/MSc.-Thesis/GameInputs/KingBase2016-03-C20-C59.pgn")

first_game = chess.pgn.read_game(pgn)

while first_game: 
    game_name = first_game.headers['White'] + '-' + first_game.headers['Black']
    out = open('ParsedGames/C2059/'+game_name+'.pgn', 'w')
    print(game_name)
    exporter = chess.pgn.FileExporter(out)
    first_game.accept(exporter)
    first_game = chess.pgn.read_game(pgn)

