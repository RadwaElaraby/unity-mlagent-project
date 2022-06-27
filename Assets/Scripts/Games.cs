using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Game0_7x7_Single_Agent : MonoBehaviour
{
    public static bool FINITE_EPISODE = true;

    public static int MAX_ENVIRONMENT_STEPS = 500; // 500

    public static int MAP_HEIGHT = 7;
    public static int MAP_WIDTH = 7;
    public static int NUM_AGENTS = 1;
    public static int SPACE_SIZE = 151;

    public static int WALL_ID = 1;
    public static int FLOOR_TILE_ID = 2;
    public static int TARGET_TILE1_ID = 201;
    public static int AGENT1_ID = 101;

    public static Vector3 CAMERA_POSITION = new Vector3(3f, 8f, -3.3f);

    public static int[] TARGET_TILES_IDs = new int[] {
        TARGET_TILE1_ID,
    };

    public static int[] AGENTS_IDs = new int[] {
        AGENT1_ID,
    };

    public static Dictionary<int, AgentInformation> agent_information = new Dictionary<int, AgentInformation>() {
        { AGENT1_ID, new AgentInformation{ order = 0, color = Color.red, target_tile_id = TARGET_TILE1_ID } },
    };

    public static Dictionary<int, TargetInformation> target_information = new Dictionary<int, TargetInformation>() {
        { TARGET_TILE1_ID, new TargetInformation{ color = Color.red, agent_id = AGENT1_ID } },
    };

    public static int[,] map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE1_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };

    public static int[,] map_walls = new int[,] {
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,         WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             0,             0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,        WALL_ID,        WALL_ID },
        };

    public static int[,] map_agents = new int[,] {
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       AGENT1_ID,       0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
        };
}




public class Game0_7x7_Static : MonoBehaviour
{
    public static bool FINITE_EPISODE = true;

    public static int MAX_ENVIRONMENT_STEPS = 500; // 500

    public static int MAP_HEIGHT = 7;
    public static int MAP_WIDTH = 7;
    public static int NUM_AGENTS = 2;
    public static int SPACE_SIZE = 306;

    public static int WALL_ID = 1;

    public static int FLOOR_TILE_ID = 2;
    public static int TARGET_TILE1_ID = 201;
    public static int TARGET_TILE2_ID = 202;

    public static int AGENT1_ID = 101;
    public static int AGENT2_ID = 102;

    public static Vector3 CAMERA_POSITION = new Vector3(3f, 8f, -3.3f);

    public static int[] TARGET_TILES_IDs = new int[] {
        TARGET_TILE1_ID,
        TARGET_TILE2_ID,
    };

    public static int[] AGENTS_IDs = new int[] {
        AGENT1_ID,
        AGENT2_ID,
    };

    public static Dictionary<int, AgentInformation> agent_information = new Dictionary<int, AgentInformation>() {
        { AGENT1_ID, new AgentInformation{ order = 0, color = Color.red, target_tile_id = TARGET_TILE1_ID } },
        { AGENT2_ID, new AgentInformation{ order = 1, color = Color.green, target_tile_id = TARGET_TILE2_ID } },
    };

    public static Dictionary<int, TargetInformation> target_information = new Dictionary<int, TargetInformation>() {
        { TARGET_TILE1_ID, new TargetInformation{ color = Color.red, agent_id = AGENT1_ID } },
        { TARGET_TILE2_ID, new TargetInformation{ color = Color.green, agent_id = AGENT2_ID } },
    };

    public static int[,] map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, TARGET_TILE2_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE1_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };

    public static int[,] map_walls = new int[,] {
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,         WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             0,             0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,        WALL_ID,        WALL_ID },
        };

    public static int[,] map_agents = new int[,] {
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       AGENT1_ID,       0,             0,             0,             AGENT2_ID,       0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
        };
}

public class Game0_7x7_Dynamic : MonoBehaviour
{
    public static bool FINITE_EPISODE = true;

    public static int MAX_ENVIRONMENT_STEPS = 100; // 500

    public static int MAP_HEIGHT = 7;
    public static int MAP_WIDTH = 7;
    public static int NUM_AGENTS = 2;
    public static int SPACE_SIZE = 306;

    public static int WALL_ID = 1;

    public static int FLOOR_TILE_ID = 2;
    public static int TARGET_TILE1_ID = 201;
    public static int TARGET_TILE2_ID = 202;

    public static int AGENT1_ID = 101;
    public static int AGENT2_ID = 102;

    public static Vector3 CAMERA_POSITION = new Vector3(3f, 8f, -3.3f);

    public static int[] TARGET_TILES_IDs = new int[] {
        TARGET_TILE1_ID,
        TARGET_TILE2_ID,
    };

    public static int[] AGENTS_IDs = new int[] {
        AGENT1_ID,
        AGENT2_ID,
    };

    public static Dictionary<int, AgentInformation> agent_information = new Dictionary<int, AgentInformation>() {
        { AGENT1_ID, new AgentInformation{ order = 0, color = Color.red, target_tile_id = TARGET_TILE1_ID } },
        { AGENT2_ID, new AgentInformation{ order = 1, color = Color.green, target_tile_id = TARGET_TILE2_ID } },
    };

    public static Dictionary<int, TargetInformation> target_information = new Dictionary<int, TargetInformation>() {
        { TARGET_TILE1_ID, new TargetInformation{ color = Color.red, agent_id = AGENT1_ID } },
        { TARGET_TILE2_ID, new TargetInformation{ color = Color.green, agent_id = AGENT2_ID } },
    };

    public static int[,] map_walls = new int[,] {
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,         WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             0,             0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,        WALL_ID,        WALL_ID },
        };

    public static int[,] map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, TARGET_TILE2_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE1_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };

    public static int[,] empty_map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };


    public static int[,] map_agents = new int[,] {
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       AGENT1_ID,       0,             0,             0,             AGENT2_ID,       0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
        };

    public static int[,] empty_map_agents = new int[,] {
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
        };

    public static int[,] map_unavailable_as_target = new int[,] {
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               1,             1,             1,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
        };
}

public class Game0_7x7_Multi_Dynamic : MonoBehaviour
{
    public static bool FINITE_EPISODE = false;

    public static int MAX_ENVIRONMENT_STEPS = 1500; // 500

    public static int MAP_HEIGHT = 7;
    public static int MAP_WIDTH = 7;
    public static int NUM_AGENTS = 2;
    public static int SPACE_SIZE = 306;

    public static int WALL_ID = 1;

    public static int FLOOR_TILE_ID = 2;
    public static int TARGET_TILE1_ID = 201;
    public static int TARGET_TILE2_ID = 202;

    public static int AGENT1_ID = 101;
    public static int AGENT2_ID = 102;

    public static Vector3 CAMERA_POSITION = new Vector3(3f, 8f, -3.3f);

    public static int[] TARGET_TILES_IDs = new int[] {
        TARGET_TILE1_ID,
        TARGET_TILE2_ID,
    };

    public static int[] AGENTS_IDs = new int[] {
        AGENT1_ID,
        AGENT2_ID,
    };

    public static Dictionary<int, AgentInformation> agent_information = new Dictionary<int, AgentInformation>() {
        { AGENT1_ID, new AgentInformation{ order = 0, color = Color.red, target_tile_id = TARGET_TILE1_ID } },
        { AGENT2_ID, new AgentInformation{ order = 1, color = Color.green, target_tile_id = TARGET_TILE2_ID } },
    };

    public static Dictionary<int, TargetInformation> target_information = new Dictionary<int, TargetInformation>() {
        { TARGET_TILE1_ID, new TargetInformation{ color = Color.red, agent_id = AGENT1_ID } },
        { TARGET_TILE2_ID, new TargetInformation{ color = Color.green, agent_id = AGENT2_ID } },
    };

    public static int[,] map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, TARGET_TILE2_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE1_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };

    public static int[,] map_walls = new int[,] {
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,         WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             0,             0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             WALL_ID,       0,             0,               WALL_ID },
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,        WALL_ID,        WALL_ID },
        };

    public static int[,] map_agents = new int[,] {
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       AGENT1_ID,       0,             0,             0,             AGENT2_ID,       0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
            { 0,       0,               0,             0,             0,             0,               0 },
        };
}



public class Game1_9x9_Static : MonoBehaviour
{
    public static bool FINITE_EPISODE = true;

    public static int MAX_ENVIRONMENT_STEPS = 500; // 500

    public static int MAP_HEIGHT = 9;
    public static int MAP_WIDTH = 9;
    public static int NUM_AGENTS = 4;
    public static int SPACE_SIZE = 1012;

    public static Vector3 CAMERA_POSITION = new Vector3(4f, 10f, -4.3f);

    public static int WALL_ID = 1;

    public static int FLOOR_TILE_ID = 2;

    public static int TARGET_TILE1_ID = 201;
    public static int TARGET_TILE2_ID = 202;
    public static int TARGET_TILE3_ID = 203;
    public static int TARGET_TILE4_ID = 204;

    public static int AGENT1_ID = 101;
    public static int AGENT2_ID = 102;
    public static int AGENT3_ID = 103;
    public static int AGENT4_ID = 104;

    public static int[] TARGET_TILES_IDs = new int[] {
        TARGET_TILE1_ID,
        TARGET_TILE2_ID,
        TARGET_TILE3_ID,
        TARGET_TILE4_ID
    };

    public static int[] AGENTS_IDs = new int[] {
        AGENT1_ID,
        AGENT2_ID,
        AGENT3_ID,
        AGENT4_ID
    };

    public static Dictionary<int, AgentInformation> agent_information = new Dictionary<int, AgentInformation>() {
        { AGENT1_ID, new AgentInformation{ order = 0, color = Color.red, target_tile_id = TARGET_TILE1_ID } },
        { AGENT2_ID, new AgentInformation{ order = 1, color = Color.green, target_tile_id = TARGET_TILE2_ID } },
        { AGENT3_ID, new AgentInformation{ order = 2, color = Color.yellow, target_tile_id = TARGET_TILE3_ID } },
        { AGENT4_ID, new AgentInformation{ order = 3, color = Color.cyan, target_tile_id = TARGET_TILE4_ID } },
    };

    public static Dictionary<int, TargetInformation> target_information = new Dictionary<int, TargetInformation>() {
        { TARGET_TILE1_ID, new TargetInformation{ color = Color.red, agent_id = AGENT1_ID } },
        { TARGET_TILE2_ID, new TargetInformation{ color = Color.green, agent_id = AGENT2_ID } },
        { TARGET_TILE3_ID, new TargetInformation{ color = Color.yellow, agent_id = AGENT3_ID } },
        { TARGET_TILE4_ID, new TargetInformation{ color = Color.cyan, agent_id = AGENT4_ID } },
    };

    public static int[,] map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   TARGET_TILE3_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE4_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   TARGET_TILE2_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE1_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };

    public static int[,] map_walls = new int[,] {
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,         WALL_ID },
            { WALL_ID, WALL_ID,         0,             0,             WALL_ID,       0,             0,             WALL_ID,         WALL_ID },
            { WALL_ID, 0,               0,             0,             WALL_ID,       0,             0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             0,             0,             0,             0,             0,               WALL_ID },
            { WALL_ID, WALL_ID,         WALL_ID,       0,             WALL_ID,       0,             WALL_ID,       WALL_ID,         WALL_ID },
            { WALL_ID, 0,               0,             0,             0,             0,             0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             0,             WALL_ID,       0,             0,             0,               WALL_ID },
            { WALL_ID, WALL_ID,         0,             0,             WALL_ID,       0,             0,             WALL_ID,         WALL_ID },
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,         WALL_ID },
        };

    public static int[,] map_agents = new int[,] {
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           AGENT1_ID,         0,             0,             0,               AGENT2_ID,       0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           AGENT4_ID,         0,             0,             0,               AGENT3_ID,       0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 }
        };

}

public class Game1_9x9_Dynamic : MonoBehaviour
{
    public static bool FINITE_EPISODE = true;

    public static int MAX_ENVIRONMENT_STEPS = 100; // 500

    public static int MAP_HEIGHT = 9;
    public static int MAP_WIDTH = 9;
    public static int NUM_AGENTS = 4;
    public static int SPACE_SIZE = 1012;

    public static Vector3 CAMERA_POSITION = new Vector3(4f, 10f, -4.3f);

    public static int WALL_ID = 1;

    public static int FLOOR_TILE_ID = 2;

    public static int TARGET_TILE1_ID = 201;
    public static int TARGET_TILE2_ID = 202;
    public static int TARGET_TILE3_ID = 203;
    public static int TARGET_TILE4_ID = 204;

    public static int AGENT1_ID = 101;
    public static int AGENT2_ID = 102;
    public static int AGENT3_ID = 103;
    public static int AGENT4_ID = 104;

    public static int[] TARGET_TILES_IDs = new int[] {
        TARGET_TILE1_ID,
        TARGET_TILE2_ID,
        TARGET_TILE3_ID,
        TARGET_TILE4_ID
    };

    public static int[] AGENTS_IDs = new int[] {
        AGENT1_ID,
        AGENT2_ID,
        AGENT3_ID,
        AGENT4_ID
    };

    public static Dictionary<int, AgentInformation> agent_information = new Dictionary<int, AgentInformation>() {
        { AGENT1_ID, new AgentInformation{ order = 0, color = Color.red, target_tile_id = TARGET_TILE1_ID } },
        { AGENT2_ID, new AgentInformation{ order = 1, color = Color.green, target_tile_id = TARGET_TILE2_ID } },
        { AGENT3_ID, new AgentInformation{ order = 2, color = Color.yellow, target_tile_id = TARGET_TILE3_ID } },
        { AGENT4_ID, new AgentInformation{ order = 3, color = Color.cyan, target_tile_id = TARGET_TILE4_ID } },
    };

    public static Dictionary<int, TargetInformation> target_information = new Dictionary<int, TargetInformation>() {
        { TARGET_TILE1_ID, new TargetInformation{ color = Color.red, agent_id = AGENT1_ID } },
        { TARGET_TILE2_ID, new TargetInformation{ color = Color.green, agent_id = AGENT2_ID } },
        { TARGET_TILE3_ID, new TargetInformation{ color = Color.yellow, agent_id = AGENT3_ID } },
        { TARGET_TILE4_ID, new TargetInformation{ color = Color.cyan, agent_id = AGENT4_ID } },
    };

    public static int[,] map_walls = new int[,] {
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,         WALL_ID },
            { WALL_ID, WALL_ID,         0,             0,             WALL_ID,       0,             0,             WALL_ID,         WALL_ID },
            { WALL_ID, 0,               0,             0,             WALL_ID,       0,             0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             0,             0,             0,             0,             0,               WALL_ID },
            { WALL_ID, WALL_ID,         WALL_ID,       0,             WALL_ID,       0,             WALL_ID,       WALL_ID,         WALL_ID },
            { WALL_ID, 0,               0,             0,             0,             0,             0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             0,             WALL_ID,       0,             0,             0,               WALL_ID },
            { WALL_ID, WALL_ID,         0,             0,             WALL_ID,       0,             0,             WALL_ID,         WALL_ID },
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,         WALL_ID },
        };

    public static int[,] map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   TARGET_TILE3_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE4_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   TARGET_TILE2_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE1_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };

    public static int[,] empty_map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };



    public static int[,] map_agents = new int[,] {
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           AGENT1_ID,         0,             0,             0,               AGENT2_ID,       0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           AGENT4_ID,         0,             0,             0,               AGENT3_ID,       0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 }
        };

    public static int[,] empty_map_agents = new int[,] {
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 }
        };


    public static int[,] map_unavailable_as_target = new int[,] {
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 1,             1,             1,               0,               0,               0 },
            { 0,       0,           0,                 1,             0,             1,               0,               0,               0 },
            { 0,       0,           0,                 1,             1,             1,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 }
        };

}

public class Game1_9x9_Multi_Dynamic : MonoBehaviour
{
    public static bool FINITE_EPISODE = false;

    public static int MAX_ENVIRONMENT_STEPS = 1500; // 500

    public static int MAP_HEIGHT = 9;
    public static int MAP_WIDTH = 9;
    public static int NUM_AGENTS = 4;
    public static int SPACE_SIZE = 1012;

    public static Vector3 CAMERA_POSITION = new Vector3(4f, 10f, -4.3f);

    public static int WALL_ID = 1;

    public static int FLOOR_TILE_ID = 2;

    public static int TARGET_TILE1_ID = 201;
    public static int TARGET_TILE2_ID = 202;
    public static int TARGET_TILE3_ID = 203;
    public static int TARGET_TILE4_ID = 204;

    public static int AGENT1_ID = 101;
    public static int AGENT2_ID = 102;
    public static int AGENT3_ID = 103;
    public static int AGENT4_ID = 104;

    public static int[] TARGET_TILES_IDs = new int[] {
        TARGET_TILE1_ID,
        TARGET_TILE2_ID, 
        TARGET_TILE3_ID, 
        TARGET_TILE4_ID 
    };

    public static int[] AGENTS_IDs = new int[] {
        AGENT1_ID,
        AGENT2_ID,
        AGENT3_ID, 
        AGENT4_ID 
    };

    public static Dictionary<int, AgentInformation> agent_information = new Dictionary<int, AgentInformation>() {
        { AGENT1_ID, new AgentInformation{ order = 0, color = Color.red, target_tile_id = TARGET_TILE1_ID } },
        { AGENT2_ID, new AgentInformation{ order = 1, color = Color.green, target_tile_id = TARGET_TILE2_ID } },
        { AGENT3_ID, new AgentInformation{ order = 2, color = Color.yellow, target_tile_id = TARGET_TILE3_ID } },
        { AGENT4_ID, new AgentInformation{ order = 3, color = Color.cyan, target_tile_id = TARGET_TILE4_ID } },
    };

    public static Dictionary<int, TargetInformation> target_information = new Dictionary<int, TargetInformation>() {
        { TARGET_TILE1_ID, new TargetInformation{ color = Color.red, agent_id = AGENT1_ID } },
        { TARGET_TILE2_ID, new TargetInformation{ color = Color.green, agent_id = AGENT2_ID } },
        { TARGET_TILE3_ID, new TargetInformation{ color = Color.yellow, agent_id = AGENT3_ID } },
        { TARGET_TILE4_ID, new TargetInformation{ color = Color.cyan, agent_id = AGENT4_ID } },
    };

    public static int[,] map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   TARGET_TILE3_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE4_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   TARGET_TILE2_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE1_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };

    public static int[,] map_walls = new int[,] {
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,         WALL_ID },
            { WALL_ID, WALL_ID,         0,             0,             WALL_ID,       0,             0,             WALL_ID,         WALL_ID },
            { WALL_ID, 0,               0,             0,             WALL_ID,       0,             0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             0,             0,             0,             0,             0,               WALL_ID },
            { WALL_ID, WALL_ID,         WALL_ID,       0,             WALL_ID,       0,             WALL_ID,       WALL_ID,         WALL_ID },
            { WALL_ID, 0,               0,             0,             0,             0,             0,             0,               WALL_ID },
            { WALL_ID, 0,               0,             0,             WALL_ID,       0,             0,             0,               WALL_ID },
            { WALL_ID, WALL_ID,         0,             0,             WALL_ID,       0,             0,             WALL_ID,         WALL_ID },
            { WALL_ID, WALL_ID,         WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,       WALL_ID,         WALL_ID },
        };

    public static int[,] map_agents = new int[,] {
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           AGENT1_ID,         0,             0,             0,               AGENT2_ID,       0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 },
            { 0,       0,           AGENT4_ID,         0,             0,             0,               AGENT3_ID,       0,               0 },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0 }
        };

}






public class Game2_10x10_Multi_Dynamic : MonoBehaviour
{
    public static bool FINITE_EPISODE = false;

    public static int MAX_ENVIRONMENT_STEPS = 1500; // 500

    public static int MAP_HEIGHT = 10;
    public static int MAP_WIDTH = 10;
    public static int NUM_AGENTS = 4;
    public static int SPACE_SIZE = 1240; /// (((10*10)*3 + 4 + 2 ) * 4) + 4*4

    public static Vector3 CAMERA_POSITION = new Vector3(4f, 11f, -4.8f);

    public static int WALL_ID = 1;

    public static int FLOOR_TILE_ID = 2;

    public static int TARGET_TILE1_ID = 201;
    public static int TARGET_TILE2_ID = 202;
    public static int TARGET_TILE3_ID = 203;
    public static int TARGET_TILE4_ID = 204;

    public static int AGENT1_ID = 101;
    public static int AGENT2_ID = 102;
    public static int AGENT3_ID = 103;
    public static int AGENT4_ID = 104;

    public static int[] TARGET_TILES_IDs = new int[] {
        TARGET_TILE1_ID,
        TARGET_TILE2_ID,
        TARGET_TILE3_ID,
        TARGET_TILE4_ID,
    };

    public static int[] AGENTS_IDs = new int[] {
        AGENT1_ID,
        AGENT2_ID,
        AGENT3_ID,
        AGENT4_ID,
    };

    public static Dictionary<int, AgentInformation> agent_information = new Dictionary<int, AgentInformation>() {
        { AGENT1_ID, new AgentInformation{ order = 0, color = Color.red, target_tile_id = TARGET_TILE1_ID } },
        { AGENT2_ID, new AgentInformation{ order = 1, color = Color.green, target_tile_id = TARGET_TILE2_ID } },
        { AGENT3_ID, new AgentInformation{ order = 2, color = Color.yellow, target_tile_id = TARGET_TILE3_ID } },
        { AGENT4_ID, new AgentInformation{ order = 3, color = Color.cyan, target_tile_id = TARGET_TILE4_ID } },
    };

    public static Dictionary<int, TargetInformation> target_information = new Dictionary<int, TargetInformation>() {
        { TARGET_TILE1_ID, new TargetInformation{ color = Color.red, agent_id = AGENT1_ID } },
        { TARGET_TILE2_ID, new TargetInformation{ color = Color.green, agent_id = AGENT2_ID } },
        { TARGET_TILE3_ID, new TargetInformation{ color = Color.yellow, agent_id = AGENT3_ID } },
        { TARGET_TILE4_ID, new TargetInformation{ color = Color.cyan, agent_id = AGENT4_ID } },
    };

    public static int[,] map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, TARGET_TILE4_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE3_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, TARGET_TILE2_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE1_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };

    public static int[,] map_walls = new int[,] {
            { WALL_ID,    WALL_ID,  WALL_ID,     WALL_ID,       WALL_ID,    WALL_ID,     WALL_ID,    WALL_ID,   WALL_ID,     WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,       WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,       WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,       WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,       WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,       WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,       WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,       WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,       WALL_ID },
            { WALL_ID,    WALL_ID,  WALL_ID,     WALL_ID,       WALL_ID,    WALL_ID,     WALL_ID,    WALL_ID,   WALL_ID,     WALL_ID },
        };

    public static int[,] map_agents = new int[,] {
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       AGENT1_ID,   0,                 0,             0,             0,               0,               0,               AGENT2_ID,       0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       AGENT3_ID,   0,                 0,             0,             0,               0,               0,               AGENT4_ID,       0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },

        };

}

public class Game2_10x10_Dynamic : MonoBehaviour
{
    public static bool FINITE_EPISODE = true;

    public static int MAX_ENVIRONMENT_STEPS = 100; // 500

    public static int MAP_HEIGHT = 10;
    public static int MAP_WIDTH = 10;
    public static int NUM_AGENTS = 4;
    public static int SPACE_SIZE = 1240; /// (((10*10)*3 + 4 + 2 ) * 4) + 4*4

    public static Vector3 CAMERA_POSITION = new Vector3(4f, 11f, -4.8f);

    public static int WALL_ID = 1;

    public static int FLOOR_TILE_ID = 2;

    public static int TARGET_TILE1_ID = 201;
    public static int TARGET_TILE2_ID = 202;
    public static int TARGET_TILE3_ID = 203;
    public static int TARGET_TILE4_ID = 204;

    public static int AGENT1_ID = 101;
    public static int AGENT2_ID = 102;
    public static int AGENT3_ID = 103;
    public static int AGENT4_ID = 104;

    public static int[] TARGET_TILES_IDs = new int[] {
        TARGET_TILE1_ID,
        TARGET_TILE2_ID,
        TARGET_TILE3_ID,
        TARGET_TILE4_ID,
    };

    public static int[] AGENTS_IDs = new int[] {
        AGENT1_ID,
        AGENT2_ID,
        AGENT3_ID,
        AGENT4_ID,
    };

    public static Dictionary<int, AgentInformation> agent_information = new Dictionary<int, AgentInformation>() {
        { AGENT1_ID, new AgentInformation{ order = 0, color = Color.red, target_tile_id = TARGET_TILE1_ID } },
        { AGENT2_ID, new AgentInformation{ order = 1, color = Color.green, target_tile_id = TARGET_TILE2_ID } },
        { AGENT3_ID, new AgentInformation{ order = 2, color = Color.yellow, target_tile_id = TARGET_TILE3_ID } },
        { AGENT4_ID, new AgentInformation{ order = 3, color = Color.cyan, target_tile_id = TARGET_TILE4_ID } },
    };

    public static Dictionary<int, TargetInformation> target_information = new Dictionary<int, TargetInformation>() {
        { TARGET_TILE1_ID, new TargetInformation{ color = Color.red, agent_id = AGENT1_ID } },
        { TARGET_TILE2_ID, new TargetInformation{ color = Color.green, agent_id = AGENT2_ID } },
        { TARGET_TILE3_ID, new TargetInformation{ color = Color.yellow, agent_id = AGENT3_ID } },
        { TARGET_TILE4_ID, new TargetInformation{ color = Color.cyan, agent_id = AGENT4_ID } },
    };

    public static int[,] map_walls = new int[,] {
            { WALL_ID,    WALL_ID,  WALL_ID,     WALL_ID,       WALL_ID,    WALL_ID,       WALL_ID,       WALL_ID,      WALL_ID,     WALL_ID },
            { WALL_ID,    0,        0,           0,             0,          0,             0,             0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           0,             0,          0,             0,             0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           0,             0,          0,             0,             0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           0,             0,          0,             0,             0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           0,             0,          0,             0,             0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           0,             0,          0,             0,             0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           0,             0,          0,             0,             0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           0,             0,          0,             0,             0,            0,           WALL_ID },
            { WALL_ID,    WALL_ID,  WALL_ID,     WALL_ID,       WALL_ID,    WALL_ID,       WALL_ID,       WALL_ID,      WALL_ID,     WALL_ID },
        };


    public static int[,] map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, TARGET_TILE4_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE3_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, TARGET_TILE2_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE1_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };

    public static int[,] empty_map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };


    public static int[,] map_agents = new int[,] {
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       AGENT1_ID,   0,                 0,             0,             0,               0,               0,               AGENT2_ID,       0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       AGENT3_ID,   0,                 0,             0,             0,               0,               0,               AGENT4_ID,       0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },

        };
    public static int[,] empty_map_agents = new int[,] {
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },

        };

    public static int[,] map_unavailable_as_target = new int[,] {
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },

        };
}

public class Game2_10x10_Dynamic_Walls : Game2_10x10_Dynamic
{
    public static int[,] map_walls = new int[,] {
            { WALL_ID,    WALL_ID,  WALL_ID,     WALL_ID,       WALL_ID,    WALL_ID,       WALL_ID,       WALL_ID,      WALL_ID,     WALL_ID },
            { WALL_ID,    0,        0,           0,             0,          0,             0,             0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           0,             0,          0,             0,             0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           WALL_ID,       0,          0,             WALL_ID,       0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           WALL_ID,       0,          0,             WALL_ID,       0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           WALL_ID,       0,          0,             WALL_ID,       0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           WALL_ID,       0,          0,             WALL_ID,       0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           0,             0,          0,             0,             0,            0,           WALL_ID },
            { WALL_ID,    0,        0,           0,             0,          0,             0,             0,            0,           WALL_ID },
            { WALL_ID,    WALL_ID,  WALL_ID,     WALL_ID,       WALL_ID,    WALL_ID,       WALL_ID,       WALL_ID,      WALL_ID,     WALL_ID },
        };
}

public class Game2_10x10_Dynamic_AllWalls : Game2_10x10_Dynamic
{
    public static int[,] map_walls = new int[,] {
            { WALL_ID,    WALL_ID,  WALL_ID,     WALL_ID,       WALL_ID,    WALL_ID,       WALL_ID,       WALL_ID,      WALL_ID,     WALL_ID },
            { WALL_ID,    0,        WALL_ID,           WALL_ID,             WALL_ID,         WALL_ID,             WALL_ID,             WALL_ID,            0,           WALL_ID },
            { WALL_ID,    WALL_ID,        WALL_ID,           WALL_ID,             WALL_ID,          WALL_ID,             WALL_ID,             WALL_ID,            WALL_ID,           WALL_ID },
            { WALL_ID,    WALL_ID,        WALL_ID,           WALL_ID,       WALL_ID,          WALL_ID,             WALL_ID,       WALL_ID,            WALL_ID,           WALL_ID },
            { WALL_ID,    WALL_ID,        WALL_ID,           WALL_ID,       WALL_ID,          WALL_ID,             WALL_ID,       WALL_ID,            WALL_ID,           WALL_ID },
            { WALL_ID,    WALL_ID,        WALL_ID,           WALL_ID,       WALL_ID,          WALL_ID,             WALL_ID,       WALL_ID,            WALL_ID,           WALL_ID },
            { WALL_ID,    WALL_ID,        WALL_ID,           WALL_ID,       WALL_ID,          WALL_ID,             WALL_ID,       WALL_ID,            WALL_ID,           WALL_ID },
            { WALL_ID,    WALL_ID,        WALL_ID,           WALL_ID,             WALL_ID,          WALL_ID,             WALL_ID,             WALL_ID,            WALL_ID,           WALL_ID },
            { WALL_ID,    0,        WALL_ID,           WALL_ID,             WALL_ID,          WALL_ID,             WALL_ID,             WALL_ID,            0,           WALL_ID },
            { WALL_ID,    WALL_ID,  WALL_ID,     WALL_ID,       WALL_ID,    WALL_ID,       WALL_ID,       WALL_ID,      WALL_ID,     WALL_ID },
        };
}



public class Game2_20x20_Multi_Dynamic : MonoBehaviour
{
    public static bool FINITE_EPISODE = false;

    public static int MAX_ENVIRONMENT_STEPS = 5000; // 500

    public static int MAP_HEIGHT = 20;
    public static int MAP_WIDTH = 20;
    public static int NUM_AGENTS = 8;
    public static int SPACE_SIZE = 9744; /// (((20*20)*3 + 8 + 2 ) * 8) + 8*8


    public static Vector3 CAMERA_POSITION = new Vector3(4f, 40f, -9.8f);

    public static int WALL_ID = 1;

    public static int FLOOR_TILE_ID = 2;

    public static int TARGET_TILE1_ID = 201;
    public static int TARGET_TILE2_ID = 202;
    public static int TARGET_TILE3_ID = 203;
    public static int TARGET_TILE4_ID = 204;
    public static int TARGET_TILE5_ID = 205;
    public static int TARGET_TILE6_ID = 206;
    public static int TARGET_TILE7_ID = 207;
    public static int TARGET_TILE8_ID = 208;

    public static int AGENT1_ID = 101;
    public static int AGENT2_ID = 102;
    public static int AGENT3_ID = 103;
    public static int AGENT4_ID = 104;
    public static int AGENT5_ID = 105;
    public static int AGENT6_ID = 106;
    public static int AGENT7_ID = 107;
    public static int AGENT8_ID = 108;

    public static int[] TARGET_TILES_IDs = new int[] {
        TARGET_TILE1_ID,
        TARGET_TILE2_ID,
        TARGET_TILE3_ID,
        TARGET_TILE4_ID,
        TARGET_TILE5_ID,        
        TARGET_TILE6_ID,
        TARGET_TILE7_ID,
        TARGET_TILE8_ID,
    };

    public static int[] AGENTS_IDs = new int[] {
        AGENT1_ID,
        AGENT2_ID,
        AGENT3_ID,
        AGENT4_ID,
        AGENT5_ID,
        AGENT6_ID,
        AGENT7_ID,
        AGENT8_ID,
    };

    public static Dictionary<int, AgentInformation> agent_information = new Dictionary<int, AgentInformation>() {
        { AGENT1_ID, new AgentInformation{ order = 0, color = Color.red, target_tile_id = TARGET_TILE1_ID } },
        { AGENT2_ID, new AgentInformation{ order = 1, color = Color.green, target_tile_id = TARGET_TILE2_ID } },
        { AGENT3_ID, new AgentInformation{ order = 2, color = Color.yellow, target_tile_id = TARGET_TILE3_ID } },
        { AGENT4_ID, new AgentInformation{ order = 3, color = Color.cyan, target_tile_id = TARGET_TILE4_ID } },
        { AGENT5_ID, new AgentInformation{ order = 4, color = Color.magenta, target_tile_id = TARGET_TILE5_ID } },
        { AGENT6_ID, new AgentInformation{ order = 5, color = Color.blue, target_tile_id = TARGET_TILE6_ID } },
        { AGENT7_ID, new AgentInformation{ order = 6, color = Color.white, target_tile_id = TARGET_TILE7_ID } },
        { AGENT8_ID, new AgentInformation{ order = 7, color = Color.gray, target_tile_id = TARGET_TILE8_ID } },
    };

    public static Dictionary<int, TargetInformation> target_information = new Dictionary<int, TargetInformation>() {
        { TARGET_TILE1_ID, new TargetInformation{ color = Color.red, agent_id = AGENT1_ID } },
        { TARGET_TILE2_ID, new TargetInformation{ color = Color.green, agent_id = AGENT2_ID } },
        { TARGET_TILE3_ID, new TargetInformation{ color = Color.yellow, agent_id = AGENT3_ID } },
        { TARGET_TILE4_ID, new TargetInformation{ color = Color.cyan, agent_id = AGENT4_ID } },
        { TARGET_TILE5_ID, new TargetInformation{ color = Color.magenta, agent_id = AGENT5_ID } },
        { TARGET_TILE6_ID, new TargetInformation{ color = Color.blue, agent_id = AGENT6_ID } },
        { TARGET_TILE7_ID, new TargetInformation{ color = Color.white, agent_id = AGENT7_ID } },
        { TARGET_TILE8_ID, new TargetInformation{ color = Color.gray, agent_id = AGENT8_ID } },
    };

    public static int[,] map_tiles = new int[,] {
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   TARGET_TILE4_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, TARGET_TILE5_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE1_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, TARGET_TILE6_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE8_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   TARGET_TILE2_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE7_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, TARGET_TILE3_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
            { FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID,   FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID, FLOOR_TILE_ID },
        };

    public static int[,] map_walls = new int[,] {
            { WALL_ID,    WALL_ID,  WALL_ID,     WALL_ID,    WALL_ID,    WALL_ID,       WALL_ID,       WALL_ID,      WALL_ID,     WALL_ID,    WALL_ID,     WALL_ID,      WALL_ID,       WALL_ID,       WALL_ID,    WALL_ID,     WALL_ID,    WALL_ID,   WALL_ID,     WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,           0,          0,           0,            0,             0,             0,          0,           0,          0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,           0,          0,           0,            0,             0,             0,          0,           0,          0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,           0,          0,           0,            0,             0,             0,          0,           0,          0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,           0,          0,           0,            0,             0,             0,          0,           0,          0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          WALL_ID,    0,             0,       0,            0,           0,          0,           0,            0,             0,             0,    0,           WALL_ID,    0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          WALL_ID,    0,             0,       0,            0,           0,          0,           0,            0,             0,             0,    0,           WALL_ID,    0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          WALL_ID,    0,             0,       0,            0,           0,          0,           0,            0,             0,             0,    0,           WALL_ID,    0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          WALL_ID,    0,             0,       0,            0,           0,          0,           0,            0,             0,             0,    0,           WALL_ID,    0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,           0,          0,           0,            0,             0,             0,          0,           0,          0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            WALL_ID,     WALL_ID,    WALL_ID,     WALL_ID,      WALL_ID,       0,             0,          0,           0,          0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,           0,          0,           0,            0,             0,             0,          0,           0,          0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          WALL_ID,    0,             0,       0,            0,           0,          0,           0,            9,             0,             0,    0,           WALL_ID,    0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          WALL_ID,    0,             0,       0,            0,           0,          0,           0,            9,             0,             0,    0,           WALL_ID,    0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          WALL_ID,    0,             0,       0,            0,           0,          0,           0,            0,             0,             0,    0,           WALL_ID,    0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          WALL_ID,    0,             0,       0,            0,           0,          0,           0,            0,             0,             0,    0,           WALL_ID,    0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,           0,          0,           0,            0,             0,             0,          0,           0,          0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,           0,          0,           0,            0,             0,             0,          0,           0,          0,         0,           WALL_ID },
            { WALL_ID,    0,        0,           0,          0,          0,             0,             0,            0,           0,          0,           0,            0,             0,             0,          0,           0,          0,         0,           WALL_ID },
            { WALL_ID,    WALL_ID,  WALL_ID,     WALL_ID,    WALL_ID,    WALL_ID,       WALL_ID,       WALL_ID,      WALL_ID,     WALL_ID,    WALL_ID,     WALL_ID,      WALL_ID,       WALL_ID,       WALL_ID,    WALL_ID,     WALL_ID,    WALL_ID,   WALL_ID,     WALL_ID },
        };

    public static int[,] map_agents = new int[,] {
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       AGENT4_ID,   0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           AGENT1_ID,         0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               AGENT5_ID,       0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             AGENT7_ID,     0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               AGENT2_ID,       0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       9,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               AGENT6_ID,       0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       AGENT3_ID,   0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 AGENT8_ID,     0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },
            { 0,       0,           0,                 0,             0,             0,               0,               0,           0,       0,          0,       0,           0,                 0,             0,             0,               0,               0,               0,               0,  },

        };

}