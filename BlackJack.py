import random
from os import system, name


class BlackJack():

  # Lets make a nice clear of the console
  def clear():
   # for windows
   if name == 'nt':
      _ = system('cls')

   # for mac and linux
   else:
      _ = system('clear')

  # Lets get all the cards values
  def cardDeal():
     # Get random cards
     card =  list(range(2,10))
     card.extend([10, 10, 10, 10, 11])
     mixit = random.choice(card)
     return (mixit)

  # calculate  total value of cards
  def scoreCards(cards):
     if sum(cards) == 21 and len(cards) == 2:
        # Thats a black jack
        return 21    
     # if above 21 remove
     if 11 in cards and sum(cards) > 21:
        cards.remove(11)
        cards.add(1)  
     return sum(cards)

 # Compare player and dealers card 
  def compare(player, dealer):
     if player == dealer:
        return "Its a drawn"
     elif dealer == 0:
        return "Dealer has BlackJack, You are the looser"
     elif player == 0:
        return "YOU ARE THE WINNER, congrats!"
     elif player > 21:
        return "OMG, pratice on your pokerface, YOU LOSE!"
     elif dealer > 21:
        return " Dealer is over 21, YOU WIN!"
     elif player > dealer:
        return "You got better score than dealer, YOU win"
     else:
         return "You loose"

   # Lets create the PLAY

   

   # create our lists to store score
  player = []
  dealer = []
  game_over = False

  player.append(cardDeal())
  player.append(cardDeal())
  dealer.append(cardDeal())

# Lets do a play for the player
  while not game_over:
     clear()
    
     player_score = scoreCards(player)
     dealer_score = scoreCards(dealer)

     print("\n########## BLACK JACK ############\n")  
    
     print("PLAYER:") 
     print(f"\nYour Cards: {player}, Current Score: {player_score}" )

     print("\nDEALER:\n")
     print(f"Dealer's first card is: {dealer[0]}")

     if player_score == 0 or player_score > 21:
        game_over = True
     else:
        print("\nDEALER talking to you so listen and think carefully about your next move:")
        lets_continue = input("  - Do you want another card? yes or no: ")
        if lets_continue == "yes":
           player.append(cardDeal())
        else:
           game_over = True
           
           
# Lest do a play for the dealer
  while dealer_score !=0 and dealer_score < 17:
     dealer.append(cardDeal())
     dealer_score = scoreCards(dealer)

  clear()
  print("\nFINAL SCORE\n")
  
  print(f"\nYour Final Hand: {player}, final score: {player_score}\n")
  
  print(f"\nDealer's final Hand: {dealer}, final score: {dealer_score}\n") 
  print(compare(player_score, dealer_score))


  print("\nGAME IS OVER, EXIT\n")


if __name__=='__main__':
    BlackJack()