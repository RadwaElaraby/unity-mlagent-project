using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Tile : MonoBehaviour
{
    public int row;
    public int col;

    public float posX;
    public float posY;
    public float posZ;


    [HideInInspector]
    public int agent_id;
    [HideInInspector]
    public int id;


    public Tile(int row, int col, float posX, float posY, float posZ)
    {
        this.row = row;
        this.col = col;

        this.posX = posX;
        this.posY = posY;
        this.posZ = posZ;
    }

}
