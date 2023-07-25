library(southparkr)
library(dplyr)

episode_list <- fetch_episode_list()

episode_lines <- fetch_all_episodes(episode_list)


# now we have a data frame with all the lines from all the episodes we want to create a corpus
# we want to merge all the lines from each episode into a single string with the chracater name



corpus <- episode_lines |>
    mutate(line = paste0(character, ": ", text))  |>
    group_by(episode_name) |>
    filter(!row_number() == 1) |> #remove first line of each episode BECAUSE IT IS THE TITLE
    filter(!row_number() == n()) |> #remove last line of each episode BECAUSE IT IS THE CREDITS
    summarise(corpus = paste0(line, collapse = "\n \n")) |>
    mutate(corpus = paste0("Episode - ",  episode_name, " \n \n \n \n", corpus, "\n \nEnd of the episode. \n \n")) #


# now we have a corpus with all the lines from all the episodes we want to merge all the lines from each episode into a single string  and save it as a txt file
corpus |> 
    pull(corpus) |>
    paste0(collapse = "\n \n \n") |>
    writeLines("southpark_corpus.txt")



