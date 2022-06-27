using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// non state
public class FloorHelper : MonoBehaviour
{
    Settings settings;

    public FloorHelper(Settings settings)
    {
        this.settings = settings;
    }

    //public int[,] MAP_TILES = new int[Game0_7x7.MAP_HEIGHT, Game0_7x7.MAP_WIDTH];
    //public Tile[,] TILES;


    public void ChangeTargetTileLocation(MyAgent agent)
    {
        var tile_id = agent.target_tile_id;
        var agent_id = agent.id;

        // choose empty tile as a new target
        int new_row;
        int new_col;
        do
        {
            new_row = UnityEngine.Random.Range(0, settings.MAP_HEIGHT-1);
            new_col = UnityEngine.Random.Range(0, settings.MAP_WIDTH-1);

        } while ((settings.MAP_WALLS[new_row, new_col] == Settings.WALL_ID) 
            || (settings.TILES[new_row, new_col].id != Settings.FLOOR_TILE_ID)
            );

        var potential_existing_agent = settings.AGENTS[new_row, new_col];
        
        // obtain the game object corresponding to the new target tile
        var another_floor_tile = settings.TILES[new_row, new_col];

        // obtain the game object corresponding to the current target tile
        var target_information = settings.TARGET_INFORMATION[tile_id];
        var target_tile = target_information.tile;

        //
        // switch information of the two tiles
        //
        var new_posX = another_floor_tile.posX;
        var new_posY = another_floor_tile.posY;
        var new_posZ = another_floor_tile.posZ;
        var new_location = another_floor_tile.transform.position;

        // switch basic info
        another_floor_tile.row = target_tile.row;
        another_floor_tile.col = target_tile.col;
        another_floor_tile.posX = target_tile.posX;
        another_floor_tile.posY = target_tile.posY;
        another_floor_tile.posZ = target_tile.posZ;

        target_tile.row = new_row;
        target_tile.col = new_col;
        target_tile.posX = new_posX;
        target_tile.posY = new_posY;
        target_tile.posZ = new_posZ;

        // switch location
        another_floor_tile.transform.position = target_tile.transform.position;
        target_tile.transform.position = new_location;

        // update trackinig info about tiles
        //settings.MAP_TILES[another_floor_tile.row, another_floor_tile.col] = Settings.FLOOR_TILE_ID;
        //settings.MAP_TILES[target_tile.row, target_tile.col] = tile_id;
        
        settings.TILES[another_floor_tile.row, another_floor_tile.col] = another_floor_tile;
        settings.TILES[target_tile.row, target_tile.col] = target_tile;

        if (potential_existing_agent != null)
        {
            potential_existing_agent.onTile = agent.onTile;
        }

        agent.onTile = another_floor_tile;
    }

    public void BuildTiles(Transform parent)
    {
        var floorTileLenX = settings.FLOOR_PREFAB.transform.localScale.x;
        var floorTileLenZ = settings.FLOOR_PREFAB.transform.localScale.z;

        var initX = 0f;
        var initZ = 0f;

        for (int row = 0; row < settings.MAP_HEIGHT; row++)
        {
            for (int col = 0; col < settings.MAP_WIDTH; col++)
            {
                var posX = initX + col * floorTileLenX;
                var posY = 0.5f;
                var posZ = initZ - row * floorTileLenZ;

                var current_id = settings.MAP_TILES[row, col];

                // add target tile if specified
                if (Array.IndexOf(Settings.TARGET_TILES_IDs, current_id) != -1)
                {
                    var tileObj = Instantiate(settings.TARGET_PREFAB, new Vector3(posX, posY, posZ), Quaternion.identity, parent);
                    var tile = tileObj.GetComponent<TargetTile>();
                    tile.row = row;
                    tile.col = col;
                    tile.posX = posX;
                    tile.posY = posY;
                    tile.posZ = posZ;
                    tile.id = current_id;

                    var target_information = settings.TARGET_INFORMATION[current_id];
                    target_information.tile = tile;
                    tile.agent_id = target_information.agent_id;

                    var tileRenderer = tileObj.GetComponent<Renderer>();
                    tileRenderer.material.color = target_information.color;

                    settings.TILES[row, col] = tile;
                }
                // otherwise, add regular tile
                else 
                {
                    var tileObj = Instantiate(settings.FLOOR_PREFAB, new Vector3(posX, posY, posZ), Quaternion.identity, parent);
                    var tile = tileObj.GetComponent<FloorTile>();
                    tile.row = row;
                    tile.col = col;
                    tile.posX = posX;
                    tile.posY = posY;
                    tile.posZ = posZ;
                    tile.id = current_id;

                    settings.TILES[row, col] = tile;
                }
            }
        }
    }

    public void BuildWalls(Transform parent)
    {
        var floorTileLenX = settings.FLOOR_PREFAB.transform.localScale.x;
        var floorTileLenZ = settings.FLOOR_PREFAB.transform.localScale.z;

        var initX = 0f;
        var initZ = 0f;

        for (int row = 0; row < settings.MAP_HEIGHT; row++)
        {
            for (int col = 0; col < settings.MAP_WIDTH; col++)
            {
                var posX = initX + col * floorTileLenX;
                var posY = 0.5f;
                var posZ = initZ - row * floorTileLenZ;

                if (settings.MAP_WALLS[row, col] == Settings.WALL_ID)
                {
                    var wallObj = Instantiate(settings.WALL_PREFAB, new Vector3(posX, posY + 1, posZ), Quaternion.identity, parent);
                    var wall = wallObj.GetComponent<Wall>();
                    settings.WALLS[row, col] = wall;
                }
            }
        }
    }

    public Tile up(Tile current)
    {
        if (current.row - 1 < 0)
            throw new OutOfAreaException();

        return settings.TILES[current.row - 1, current.col]; 
    }
    public Tile down(Tile current)
    {
        if (current.row + 1 >= settings.MAP_HEIGHT)
            throw new OutOfAreaException();

        return settings.TILES[current.row + 1, current.col];
    }
    public Tile right(Tile current)
    {
        if (current.col + 1 >= settings.MAP_WIDTH)
            throw new OutOfAreaException();

        return settings.TILES[current.row, current.col + 1];
    }
    public Tile left(Tile current)
    {
        if (current.col - 1 < 0)
            throw new OutOfAreaException();

        return settings.TILES[current.row, current.col - 1];
    }

    
}
