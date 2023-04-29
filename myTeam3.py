# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
from util import PriorityQueue
import random
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'offenseAgent', second = 'mirrorDefenseAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class offenseAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.food = len(self.getFood(gameState).asList()) # holds number of food before crossing over
    self.stuck_counter = 0 # initialize stuck counter to 0
    self.last_position = (0,0)
    if self.red:
      self.ourSideCenter = gameState.data.layout.width // 2 - 1
      self.theirSideCenter = gameState.data.layout.width // 2
      self.enemyOffenseAgent = 1
    else:
      self.enemyOffenseAgent = 0
      self.ourSideCenter = gameState.data.layout.width // 2 
      self.theirSideCenter = gameState.data.layout.width // 2 - 1
    self.entrances = self.findEntrances(gameState)
    self.up = True
    self.targetCap = (0,0)
    self.capsuleTarget = False 
    self.listCloser = []
    self.Escape = False
    self.targetRoam = None
    self.food = len(self.getFood(gameState).asList())
    #self.debugDraw((16,7), (100,100,100))

  ############################
  ## BFS to target position ##
  ############################
  def isGoal2(self,successor, targetPosition):
    if successor.getAgentState(self.index).getPosition() == targetPosition:
      return True
    return False

  def bfs2(self, gameState, actions, targetPos):
    def heuristic(gameState2):
      #print(targetPos)
      #print("Heuristic ,",self.distancer.getDistance(targetPos, gameState2.getAgentState(self.index).getPosition()))
      return self.distancer.getDistance(targetPos, gameState2.getAgentState(self.index).getPosition())

    solution = [] #cannot hardcode this because it might be illegal. figure out better solution the problem is sometimes the stack empties out and nothing is returned
    stack = util.PriorityQueue()
    counter = util.Counter()
    if self.isGoal2(gameState, targetPos):
      return (Directions.STOP, False)
    for a in actions:
      stack.push((gameState.generateSuccessor(self.index,a),a),heuristic(gameState.generateSuccessor(self.index,a)))
    while stack.isEmpty() is not True:

        node = stack.pop() # stack pops a state and action list and visited list
        if counter[node[0].getAgentState(self.index).getPosition()] == 1:
            continue
        ##print(node[0].getAgentState(self.index).getPosition())
        if self.isGoal2(node[0],targetPos):
            solution = node[1]
            break
        counter[node[0].getAgentState(self.index).getPosition()] = 1
        children = []  

        self.debugDraw(node[0].getAgentPosition(self.index), (100,200,150))
        for a in node[0].getLegalActions(self.index):
          children.append((node[0].generateSuccessor(self.index,a),node[1]))# each child is a gamestate
        for child in children:
          stack.push(child,heuristic(child[0]))
    if solution == []:
      #print("random")
      return  (random.choice(actions), False)
    ##print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    return (solution, True)

####################################
#Second BFS for roaming on our side#
####################################
  def bfs2Roam(self, gameState, actions, targetPos):
    def heuristic(gameState2):
      #print("Heuristic ,",self.distancer.getDistance(targetPos, gameState2.getAgentState(self.index).getPosition()))
      return self.distancer.getDistance(targetPos, gameState2.getAgentState(self.index).getPosition())

    solution = [] #cannot hardcode this because it might be illegal. figure out better solution the problem is sometimes the stack empties out and nothing is returned
    stack = util.PriorityQueue()
    counter = util.Counter()
    if self.isGoal2(gameState, targetPos):
      return (Directions.STOP, False)
    for a in actions:
      stack.push((gameState.generateSuccessor(self.index,a),a),heuristic(gameState.generateSuccessor(self.index,a)))
    while stack.isEmpty() is not True:
        
        node = stack.pop() # stack pops a state and action list and visited list
        if counter[node[0].getAgentState(self.index).getPosition()] == 1:
            continue
        if node[0].getAgentState(self.index).isPacman:
          continue
        ##print(node[0].getAgentState(self.index).getPosition())
        if self.isGoal2(node[0],targetPos):
            solution = node[1]
            break
        counter[node[0].getAgentState(self.index).getPosition()] = 1
        children = []  
        #self.debugDraw(node[0].getAgentPosition(self.index), (200,100,150))
        for a in node[0].getLegalActions(self.index):
          children.append((node[0].generateSuccessor(self.index,a),node[1]))# each child is a gamestate
        for child in children:
          stack.push(child,heuristic(child[0]))
    if solution == []:
      #print("random")
      return  (random.choice(actions), False)
    ##print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    return (solution, True)



####################################
##  BFS for escaping if trapped   ##
####################################

  def isGoal3(self,successor):
    for i in self.entrances:
      if successor.getAgentState(self.index).getPosition() == (self.ourSideCenter, i):
        return True  
    return False


  def bfs3Escape(self, gameState, actions):
    def heuristic(gameState2):
      enemies = self.getEnemies(gameState)
      if enemies == []:
        return 1
     
      min = [1000,-1]
      for a in enemies:
        m = self.distancer.getDistance(gameState2.getAgentState(a).getPosition(), gameState2.getAgentState(self.index).getPosition())
        if m < min[0]:
          min = [m,a]
      en = min[1]
      return -1 * self.distancer.getDistance(gameState2.getAgentState(en).getPosition(), gameState2.getAgentState(self.index).getPosition())

    solution = [] #cannot hardcode this because it might be illegal. figure out better solution the problem is sometimes the stack empties out and nothing is returned
    stack = util.PriorityQueue()
    counter = util.Counter()
    if self.isGoal3(gameState):
      return (Directions.STOP, False)
    for a in actions:
      stack.push((gameState.generateSuccessor(self.index,a),a),heuristic(gameState.generateSuccessor(self.index,a)))
    while stack.isEmpty() is not True:
        node = stack.pop() # stack pops a state and action list and visited list
        if counter[node[0].getAgentState(self.index).getPosition()] == 1:
            continue

        ##print(node[0].getAgentState(self.index).getPosition())
        if self.isGoal3(node[0]):
            solution = node[1]
            break
        counter[node[0].getAgentState(self.index).getPosition()] = 1
        children = []  
        #self.debugDraw(node[0].getAgentPosition(self.index), (142,200,50))
        for a in node[0].getLegalActions(self.index):
          children.append((node[0].generateSuccessor(self.index,a),node[1]))# each child is a gamestate
        for child in children:
          stack.push(child,heuristic(child[0]))
    if solution == []:
      #print("random")
      return  (random.choice(actions), False)
    ##print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    return (solution, True)


#####################################################################################
##  Updates self.closerFood to closest food that u can get without being captured  ##               
#####################################################################################
  def closerFood(self,gameState):
    self.listCloser = []
    foodList = self.getFood(gameState).asList()  
    if len(foodList) <= 2 :
      return False
    enemies = self.getEnemies(gameState)
    if enemies == []:
      self.listCloser.extend(foodList)
      return True
    for food in foodList:

      enDist = [self.distancer.getDistance(gameState.getAgentPosition(a), food) for a in enemies]
      minEnemy = min(enDist)
      if(self.distancer.getDistance(gameState.getAgentPosition(self.index), food) < minEnemy):
        #print("Closeer", food)
        #self.debugDraw(food, (20,200,150))
        self.listCloser.append(food)


#################################################
## Moves up and down looking for opportunities ##
#################################################
  def roam(self, curr):
    chosen=-1
    if not self.up:
      chosen = self.entrances[len(self.entrances)-1]
      start = len(self.entrances)-1
      while start >=0 :
        if self.entrances[start] < curr:
          chosen = self.entrances[start]
          break
        start-=1
      if start == 0: 
        self.up = True
    else:
      chosen =self.entrances[0]
      start = 0
      while start <len(self.entrances) :
        if self.entrances[start] > curr:
          chosen = self.entrances[start]
          break
        start+=1   
      if start  >= len(self.entrances)-1: 
        self.up = False
    return chosen

######################################################################################
## If you are closer to a capsule, then the enemy. Return action on path to capsule ##
######################################################################################
  def capsuleAction(self,gameState, actions):
    # If enemy is feared, then don't path to capsule (Might still accidently eat it tho need to handle that)
    if gameState.getAgentState(self.getOpponents(gameState)[0]).scaredTimer > 10:
      return False

    # If just ate target capsule, then there is no target
    if self.capsuleTarget and self.targetCap not in self.getCapsules(gameState):
      self.capsuleTarget = False
    
    enemies = self.getEnemies(gameState)
    #See if you are closer than the enemy to a capsule
    if self.capsuleTarget == False:
      for cap in self.getCapsules(gameState):
        enemyDist = 1000
        if enemies !=[]:
          enemyDist = min([self.distancer.getDistance(cap, gameState.getAgentState(a).getPosition()) for a in enemies])
        myDist = self.distancer.getDistance(cap, gameState.getAgentState(self.index).getPosition())
        #print("Mydist: ",myDist,"Enemy Dist:", enemyDist)
        if not self.capsuleTarget and  myDist < enemyDist:
          self.capsuleTarget = True
          self.targetCap = cap

    if self.capsuleTarget == True:
      #self.debugDraw(self.targetCap, (120,120,200))
      ret =  self.bfs2(gameState,actions,self.targetCap) # If bfs fails, then ret[1] is false, and no path exists
      if(ret[1]== False):
        #print("Abort")
        self.capsuleTarget = False
        return ret[1]
      else:
        return ret[0]
    else:
      return False

############################################
# Returns non feared non attacking enemies #
############################################
  def getEnemies(self,gameState):
    return [i for i in self.getOpponents(gameState) if (not gameState.getAgentState(i).isPacman) and (not gameState.getAgentState(i).scaredTimer>10)and ( gameState.getAgentState(i).getPosition() != None)]
  
######################################
# Code for escaping back to our side #
######################################  
  def escapeSearch(self,gameState,actions):
    self.Escape = True
    # Find escape path
    minDistance = [1000,()]
    enemies = self.getEnemies(gameState)
    for entr in self.entrances:
      enemyDist = 1000
      if enemies!=[]:
        enemyDist = min([self.distancer.getDistance((self.ourSideCenter,entr), gameState.getAgentState(a).getPosition()) for a in self.getEnemies(gameState)])

      myDist = self.distancer.getDistance((self.ourSideCenter,entr), gameState.getAgentState(self.index).getPosition())
      if myDist < enemyDist and myDist < minDistance[0]:
        minDistance = [myDist,entr]
        
    if(minDistance[0] != 1000):# There is an escape nearby so bfs2 to it
      target = (self.ourSideCenter,minDistance[1])
    else: 
      ret = self.bfs3Escape(gameState,actions) # no escape nearby, so bfs3 escape
      if ret[1] == False:
        pass
        #print("TRAPPED FFS")
      #print("Escape search")
      return ret[0] # Bad. Should search for escape using another search

    ret = self.bfs2(gameState,actions,target)
    if ret[1] == False: # BFS to target failed. Should never happen because maze distance is less than enemies
      #print("UNLUCKY")
      pass
    return ret[0]  
      #self.debugDraw(gameState.getAgentPosition(self.index), (120,133,121))
      
########################################
## Chooses action based on heuristics ##
########################################
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    myPos = gameState.getAgentState(self.index).getPosition()
    self.debugClear()
    if(not gameState.getAgentState(self.index).isPacman):
      self.food = len(self.getFood(gameState).asList())
    
    #If time is lo I'm a pacman and carrying food, then escape
    if(gameState.data.timeleft < 200 and gameState.getAgentState(self.index).isPacman and self.food - len(self.getFood(gameState).asList()) > 0):
      ret = self.bfs3Escape(gameState,actions)
      return ret[0]

    # Capsule Search
    capsuleMove = self.capsuleAction(gameState, actions)
    if capsuleMove != False:
      return capsuleMove

    # If I am a ghost, then i am not trying to escape. Check out the Escape code
    if(not gameState.getAgentState(self.index).isPacman):
      self.Escape = False

    # Food Search Need to make smarter
    self.closerFood(gameState)
    if(len(self.listCloser) > 0) and self.Escape == False:
      minDistance = [1000,()]
      organized = util.PriorityQueue()
      for food in self.listCloser:
        distance = self.distancer.getDistance(myPos, food) 
        organized.push((food,distance),distance)

      foodClosestEntrance = None
      target = None
      indices = self.getEnemies(gameState)
      if indices == []:
        print("they are feared")
        target, dist = organized.pop()
        ret = self.bfs2(gameState, actions, target)
        if ret[1] == False:
          # print("no path found to food")
          pass
        else:
          return ret[0]
      else:
        #Find closest enemy index
        dist = gameState.getAgentDistances() # indices give us distance
        values = [dist[a] for a in indices] 
        minValue = min(values)
        closestEnemyIndex = [a for a, v in zip(indices, values) if v == minValue] # Choose the best action
        closestEnemyIndex = random.choice(closestEnemyIndex)
        minDistance2 = 100
        itsPossible = False
        onFood = None
        while (not itsPossible) and organized.isEmpty() == False:
          onFood, dist = organized.pop() # (pos), distance
          print(dist, "To food closest")
          self.debugDraw(onFood, (50,150,24))
          # Distance from food to every entrance
          for entry in self.entrances:
            if self.getMazeDistance(onFood, (self.ourSideCenter,entry)) < minDistance2:
              minDistance2 = self.getMazeDistance(onFood, (self.ourSideCenter,entry))
              #print("Min distance: ", minDistance)
              foodClosestEntrance = (self.ourSideCenter, entry)

          #Their distance to this entrance
          # print(foodClosestEntrance, "Hi")
          # print(closestEnemyIndex, "Hi2")
          self.debugDraw(foodClosestEntrance, (230,120,200))
          theirDistancetoEntrance = self.distancer.getDistance(gameState.getAgentPosition(closestEnemyIndex), foodClosestEntrance)
          ourDistancetoEntranceFromFood = self.distancer.getDistance(onFood, foodClosestEntrance)
          ourHeuristic = 0
          theirHeuristic = 0
          if gameState.getAgentState(self.index).isPacman:
            ourHeuristic = dist*2+ourDistancetoEntranceFromFood
            theirHeuristic = theirDistancetoEntrance-1
          else:
            ourHeuristic= ourDistancetoEntranceFromFood + self.distancer.getDistance(foodClosestEntrance, onFood)
            theirHeuristic = theirDistancetoEntrance
          print("ours", ourHeuristic)
          print("Their heurisitc,",theirHeuristic)
          if ( ourHeuristic < theirHeuristic ): 
            itsPossible = True
        if itsPossible == False:
          pass
        else:
          ret = self.bfs2(gameState, actions, onFood)
          if ret[1] == False:
            # print("no path found to food")
            pass
          else:
            return ret[0]
      
    # Escape Search
    if(gameState.getAgentState(self.index).isPacman):
      return self.escapeSearch(gameState, actions)
    else:
      #Roam Code
      if myPos == self.targetRoam:
        self.targetRoam = (self.ourSideCenter, self.roam(myPos[1]))
      elif self.targetRoam == None:
        self.targetRoam = (self.ourSideCenter, self.roam(myPos[1]))
      #self.debugDraw(self.targetRoam,(100,100,200))
      
      ret = self.bfs2Roam(gameState,actions,self.targetRoam)
      return ret[0]


  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
  def evaluate2(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures2(gameState, action)
    weights = self.getWeights2(gameState, action)
    return features * weights

  def getFeatures2(self, gameState,action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    cap = self.getCapsules(gameState)
    # Compute distance to the nearest food
    if len(cap) > 0: # If there exist capsules
      myPos = successor.getAgentState(self.index).getPosition()  
      minDistance = min([self.getMazeDistance(myPos, caps) for caps in cap])
      features['distanceToCap'] = minDistance

    return features

  def getWeights2(self, gameState, action):
    return {'successorScore': 100,'distanceToCap': -1}
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)

    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()  
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100,'distanceToFood': -1}
  

  # returns a list of entrances
  def findEntrances(self, gameState):
    """
    Returns a list of entrances between the two horizontal sides of the board.
    """
    walls = gameState.getWalls()
    height = gameState.data.layout.height
    entrances = []
    for y in range(1, height - 1):
        if not walls[self.ourSideCenter][y] and not walls[self.theirSideCenter][y]:
          entrances.append(y)
    return entrances


# perfect mirror
class mirrorDefenseAgent(CaptureAgent):
  """
  First chooses a random enemy agent to mirror.
  If the enemy agent turns into a pac man, that pacman is listed as the offense agent and is mirrored now
  """
  def registerInitialState(self, gameState):
      CaptureAgent.registerInitialState(self, gameState)
      self.start = gameState.getAgentPosition(self.index)
      if self.red:
        self.ourSideCenter = gameState.data.layout.width // 2 - 1
        self.theirSideCenter = gameState.data.layout.width // 2
        self.enemyOffenseAgent = 1
      else:
        self.enemyOffenseAgent = 0
        self.ourSideCenter = gameState.data.layout.width // 2 
        self.theirSideCenter = gameState.data.layout.width // 2 - 1
      self.entrances = self.findEntrances(gameState)
      self.enemyClosestEntrance = self.entrances[0]
      # our side center and their side center for entrance calculations
      ##print("Entrances: ", self.entrances)
      for entrance in self.entrances:
        self.debugDraw((self.ourSideCenter, entrance), (200, 200, 200))
        self.debugDraw((self.theirSideCenter, entrance), (100, 100, 200))

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index) # south north ease est stop
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemiesIndexes = [i for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = []
    if len(invaders) > 0:

      # ensure you have the right enemy offense agent
      # find the closer invader
      #minDistance = 100
      #closestInvader = -1
      #for invader in invaders:
      #  if self.distancer.getDistance(invader.getPosition(), gameState.getPosition()) < minDistance:
      #    minDistance = self.distancer.getDistance(invader.getPosition(), gameState.getPosition())
      #    closestInvader = invader.index        
      #self.enemyOffenseAgent = invaders[closestInvader]
      self.enemyOffenseAgent = enemiesIndexes[0]
      values = [self.evaluate(gameState, a) for a in actions]
    else: #no attackers, mirror enemy
      # determine which entrance the enemy is closest to
      minDistance = 100
      self.enemyClosestEntrance = self.entrances[0]
      for entry in self.entrances:
        if self.getMazeDistance(gameState.getAgentState(self.enemyOffenseAgent).getPosition(), (self.theirSideCenter,entry)) < minDistance:
          minDistance = self.getMazeDistance(gameState.getAgentState(self.enemyOffenseAgent).getPosition(), (self.theirSideCenter,entry))
          ##print("Min distance: ", minDistance)
          self.enemyClosestEntrance = entry

      return self.bfs(gameState, actions)
      values = [self.evaluate2(gameState, a) for a in actions] # Given legal actions. Evaluate each of them [10,30,-1300,1,3]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue] # Choose the best action
    foodLeft = len(self.getFood(gameState).asList()) 

        
    
    if foodLeft <= 2:#Can food ever be less than 2?
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions) #given 2 or more moves that have the same heuristic value choose one randomly

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
    
    
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """

    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights


  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1


    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

  def evaluate2(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures2(gameState, action)
    weights = self.getWeights2(gameState, action)
    return features * weights

  #Stay near food close to enemy 
  def getFeatures2(self, gameState, action):
    #Useful variables
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    foodList = self.getFoodYouAreDefending(successor).asList()    
    enemies = self.getOpponents(successor) # enemies indices in list
    

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # distance to their closest entrance
    features['distanceToEnemyEntrance'] = self.getMazeDistance(successor.getAgentState(self.index).getPosition(), (self.ourSideCenter, self.enemyClosestEntrance))
    

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    return features

  def getWeights2(self, gameState, action):
    return {'onDefense': 0, 'distanceToEnemyEntrance': -1}

  # returns a list of entrances
  def findEntrances(self, gameState):
    """
    Returns a list of entrances between the two horizontal sides of the board.
    """
    walls = gameState.getWalls()
    height = gameState.data.layout.height
    entrances = []
    for y in range(1, height - 1):
        if not walls[self.ourSideCenter][y] and not walls[self.theirSideCenter][y]:
          entrances.append(y)
    return entrances
  def bfs(self,gameState,actions):
    solution = [] #cannot hardcode this because it might be illegal. figure out better solution the problem is sometimes the stack empties out and nothing is returned
    stack = util.Queue()
    counter = util.Counter()
    if self.isGoal(gameState):
      return Directions.STOP
    for a in actions:
      stack.push((gameState.generateSuccessor(self.index,a),a,a))
    while stack.isEmpty() is not True:
        node = stack.pop() # stack pops a state and action list and visited list
        if counter[node[0].getAgentState(self.index).getPosition()] == 1:
            continue
        if node[0].getAgentState(self.index).isPacman:
          continue

        if self.isGoal(node[0]):
            solution = node[1]
            break
        counter[node[0].getAgentState(self.index).getPosition()] = 1
        children = []
        for a in node[0].getLegalActions(self.index):
          children.append((node[0].generateSuccessor(self.index,a),node[1],a))# each child is a gamestate
        for child in children:
          stack.push(child)
    if solution == []:
      self.offense = False
      return  random.choice(actions)

    self.offense = False
    return solution

  # goal state is when you are mirroring their offense agent on their closest entrance
  def isGoal(self, gameState):
    previousState = self.getPreviousObservation()
    if gameState.getAgentState(self.index).getPosition() == (self.ourSideCenter, self.enemyClosestEntrance): #goal is to match them
      return True
    return False

  def bfs2(self,gameState,actions):
    solution = [] #cannot hardcode this because it might be illegal. figure out better solution the problem is sometimes the stack empties out and nothing is returned
    stack = util.Queue()
    counter = util.Counter()
    if self.isGoal(gameState):
      return Directions.STOP
    for a in actions:
      stack.push((gameState.generateSuccessor(self.index,a),a,a))
    while stack.isEmpty() is not True:
        node = stack.pop() # stack pops a state and action list and visited list
        if counter[node[0].getAgentState(self.index).getPosition()] == 1:
            continue
        if node[0].getAgentState(self.index).isPacman:
          continue

        if self.isGoal(node[0]):
            solution = node[1]
            break
        counter[node[0].getAgentState(self.index).getPosition()] = 1
        children = []
        for a in node[0].getLegalActions(self.index):
          children.append((node[0].generateSuccessor(self.index,a),node[1],a))# each child is a gamestate
        for child in children:
          stack.push(child)
    if solution == []:
      self.offense = False
      return  random.choice(actions)

    self.offense = False
    return solution

  # goal state is when you are mirroring their offense agent on their closest entrance
  def isGoal2(self, gameState):
    previousState = self.getPreviousObservation()
    if gameState.getAgentState(self.index).getPosition() == (self.ourSideCenter, self.enemyClosestEntrance): #goal is to match them
      return True
    return False
