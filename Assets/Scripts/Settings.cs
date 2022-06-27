using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// state class
public class Settings : MonoBehaviour
{
    // Start is called before the first frame update
    public GameObject FLOOR_PREFAB;
    public GameObject TARGET_PREFAB;
    public GameObject AGENT_PREFAB;
    public GameObject WALL_PREFAB;


    [HideInInspector]
    public const int NO_ACTION = 0;
    [HideInInspector]
    public const int UP_ACTION = 1;
    [HideInInspector]
    public const int DOWN_ACTION = 2;
    [HideInInspector]
    public const int RIGHT_ACTION = 3;
    [HideInInspector]
    public const int LEFT_ACTION = 4;

    [HideInInspector]
    public static int ACTION_SPACE_PER_AGENT = 1;


    [HideInInspector]
    public static bool FINITE_EPISODE = Game2_10x10_Dynamic_Walls.FINITE_EPISODE;
    [HideInInspector]
    public static int FLOOR_TILE_ID = Game2_10x10_Dynamic_Walls.FLOOR_TILE_ID;
    [HideInInspector]
    public static int WALL_ID = Game2_10x10_Dynamic_Walls.WALL_ID;

    public int NUM_AGENTS;
    public int MAP_WIDTH;
    public int MAP_HEIGHT;

    public static int[] TARGET_TILES_IDs = Game2_10x10_Dynamic_Walls.TARGET_TILES_IDs;
    public static int[] AGENTS_IDs = Game2_10x10_Dynamic_Walls.AGENTS_IDs;

    public Dictionary<int, AgentInformation> AGENT_INFORMATION = Game2_10x10_Dynamic_Walls.agent_information;
    public Dictionary<int, TargetInformation> TARGET_INFORMATION = Game2_10x10_Dynamic_Walls.target_information;

    public int[,] MAP_TILES = new int[Game2_10x10_Dynamic_Walls.MAP_HEIGHT, Game2_10x10_Dynamic_Walls.MAP_WIDTH];
    public int[,] MAP_WALLS = new int[Game2_10x10_Dynamic_Walls.MAP_HEIGHT, Game2_10x10_Dynamic_Walls.MAP_WIDTH];
    public int[,] MAP_AGENTS = new int[Game2_10x10_Dynamic_Walls.MAP_HEIGHT, Game2_10x10_Dynamic_Walls.MAP_WIDTH];
    public int[,] MAP_UNAVAILABLE_AS_TARGET = new int[Game2_10x10_Dynamic_Walls.MAP_HEIGHT, Game2_10x10_Dynamic_Walls.MAP_WIDTH];

    [HideInInspector]
    public static int MAX_ENVIRONMENT_STEPS = Game2_10x10_Dynamic_Walls.MAX_ENVIRONMENT_STEPS;

    public Tile[,] TILES;
    public Wall[,] WALLS;
    public MyAgent[,] AGENTS;

    public void CreateGameMap()
    {
        cleanUp();

        TILES = new Tile[MAP_HEIGHT, MAP_WIDTH];
        WALLS = new Wall[MAP_HEIGHT, MAP_WIDTH];
        AGENTS = new MyAgent[MAP_HEIGHT, MAP_WIDTH];


        MAP_TILES = Game2_10x10_Dynamic_Walls.map_tiles;
        MAP_WALLS = Game2_10x10_Dynamic_Walls.map_walls;
        MAP_AGENTS = Game2_10x10_Dynamic_Walls.map_agents;
        MAP_UNAVAILABLE_AS_TARGET = Game2_10x10_Dynamic_Walls.map_unavailable_as_target;

    }

    public void CreateRandomGameMap()
    {
        cleanUp();

        TILES = new Tile[MAP_HEIGHT, MAP_WIDTH];
        WALLS = new Wall[MAP_HEIGHT, MAP_WIDTH];
        AGENTS = new MyAgent[MAP_HEIGHT, MAP_WIDTH];

        MAP_WALLS = Game2_10x10_Dynamic_Walls.map_walls;

        // place agents
        MAP_AGENTS = (int[,])Game2_10x10_Dynamic_Walls.empty_map_agents.Clone();
        foreach (var id in Settings.AGENTS_IDs)
        {
            var agent_information = AGENT_INFORMATION[id];

            var agent = agent_information.agent;
            var agent_id = agent.id;

            // choose empty tile
            int row;
            int col;
            do
            {
                row = UnityEngine.Random.Range(0, MAP_HEIGHT - 1);
                col = UnityEngine.Random.Range(0, MAP_WIDTH - 1);

            } while (MAP_WALLS[row, col] == Settings.WALL_ID || (MAP_AGENTS[row, col] != 0));


            MAP_AGENTS[row, col] = agent_id;
        }

        // place target tiles
        MAP_TILES = (int[,])Game2_10x10_Dynamic_Walls.empty_map_tiles.Clone();
        foreach (var id in Settings.AGENTS_IDs)
        {
            var agent_information = AGENT_INFORMATION[id];

            var agent = agent_information.agent;
            var tile_id = agent.target_tile_id;

            // choose empty tile
            int row;
            int col;
            do
            {
                row = UnityEngine.Random.Range(0, MAP_HEIGHT - 1);
                col = UnityEngine.Random.Range(0, MAP_WIDTH - 1);

            } while ((MAP_WALLS[row, col] == WALL_ID) || (MAP_TILES[row, col] != FLOOR_TILE_ID) || (MAP_UNAVAILABLE_AS_TARGET[row, col] == 1));

            MAP_TILES[row, col] = tile_id;
        }
    }

    public void cleanUp()
    {
        // clean up previous area
        if (TILES != null)
        {
            for (int row = 0; row < MAP_HEIGHT; row++)
            {
                for (int col = 0; col < MAP_WIDTH; col++)
                {
                    Tile tileObj = TILES[row, col];
                    if (tileObj != null)
                    {
                        Destroy(tileObj.gameObject);
                    }
                }
            }
        }

        // clean up previous area
        if (WALLS != null)
        {
            for (int row = 0; row < MAP_HEIGHT; row++)
            {
                for (int col = 0; col < MAP_WIDTH; col++)
                {
                    Wall wallObj = WALLS[row, col];
                    if (wallObj != null)
                    {
                        Destroy(wallObj.gameObject);
                    }
                }
            }
        }

    }
    
}
