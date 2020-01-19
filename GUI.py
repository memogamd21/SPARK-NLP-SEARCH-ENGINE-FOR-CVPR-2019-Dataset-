import easygui as eg

def ask():
    msg = "Enter your query"
    title = "Query"
    response = eg.enterbox(msg, title)
    return response


def display_query_results(docs_list, query_tokens):
    msg = "Preprocessed query: "+str(query_tokens)+"\nThese are the results of your query, you can double click" \
                                                   " or select and press ok on a" \
                                                   " result to open the pdf paper in your local device" \
                                                   " or you can choose to Show more results. Press cancel to go back" \
                                                   " to the main menu."
    title = "Query results"
    choice = eg.choicebox(msg, title)
    return choice
