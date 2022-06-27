using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class OutOfAreaException : Exception { }
public class WallCollision : Exception { }
public class AgentInformation
{
    public int order;
    public MyAgent agent;
    public Color color;
    public int target_tile_id;
}

public class TargetInformation
{
    public Color color;
    public int agent_id;
    public TargetTile tile;
}