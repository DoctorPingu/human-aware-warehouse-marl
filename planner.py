import heapq
import math
from layouts import point_in_poly, polygon_edges

def _pt_seg_dist2(px,py,x1,y1,x2,y2):
    vx,vy = x2-x1,y2-y1
    L2 = vx*vx+vy*vy
    t = 0.0 if L2==0 else max(0.0,min(1.0,((px-x1)*vx+(py-y1)*vy)/L2))
    cx,cy = x1+t*vx, y1+t*vy
    return (px-cx)**2+(py-cy)**2

def _near_poly(px,py,poly,margin):
    if point_in_poly(px,py,poly):
        return True
    m2 = margin*margin
    for (x1,y1,x2,y2) in polygon_edges(poly):
        if _pt_seg_dist2(px,py,x1,y1,x2,y2) <= m2:
            return True
    return False

def seg_hits_poly(p1,p2,poly):
    def ccw(a,b,c): return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    edges = polygon_edges(poly)
    for (x1,y1,x2,y2) in edges:
        a,b,c,d = p1,p2,(x1,y1),(x2,y2)
        if ccw(a,c,d)!=ccw(b,c,d) and ccw(a,b,c)!=ccw(a,b,d):
            return True
    if point_in_poly(p1[0],p1[1],poly) or point_in_poly(p2[0],p2[1],poly):
        return True
    return False

def clear_segment(p1,p2,shelves):
    return not any(seg_hits_poly(p1,p2,poly) for poly in shelves)

def clear_segment_margin(p1,p2,shelves,margin,samples=20):
    for k in range(samples+1):
        t = k/samples
        px, py = p1[0]+(p2[0]-p1[0])*t, p1[1]+(p2[1]-p1[1])*t
        for poly in shelves:
            if _near_poly(px,py,poly,margin):
                return False
    return True

class Planner:
    def __init__(self, footprint, shelves, res=0.2, margin=0.90):
        self.footprint = footprint
        self.shelves = shelves
        self.res = res
        self.margin = margin
        xs = [p[0] for p in footprint]; ys = [p[1] for p in footprint]
        self.x0, self.x1 = min(xs), max(xs)
        self.y0, self.y1 = min(ys), max(ys)
        self.nx = int((self.x1-self.x0)/res)+1
        self.ny = int((self.y1-self.y0)/res)+1
        self._blocked_cache = {}

    def _cell(self,x,y):
        return (int(round((x-self.x0)/self.res)), int(round((y-self.y0)/self.res)))

    def _world(self,cx,cy):
        return (self.x0+cx*self.res, self.y0+cy*self.res)

    def _blocked(self, cx, cy):
        key=(cx,cy)
        v = self._blocked_cache.get(key)
        if v is not None: return v
        if not (0<=cx<self.nx and 0<=cy<self.ny):
            v = True
        else:
            wx,wy = self._world(cx,cy)
            if not point_in_poly(wx,wy,self.footprint):
                v = True
            else:
                v = any(_near_poly(wx,wy,poly,self.margin) for poly in self.shelves)
        self._blocked_cache[key]=v
        return v

    def route(self, p1, p2):
        start, goal = self._cell(*p1), self._cell(*p2)
        if self._blocked(*start): start = self._nudge(start)
        if self._blocked(*goal): goal = self._nudge(goal)

        def h(c): return math.hypot(c[0]-goal[0], c[1]-goal[1])
        openq = [(h(start), 0, start, None)]
        came = {}
        gscore = {start: 0}
        seen = set()
        neigh = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
                 (1,1,1.414),(1,-1,1.414),(-1,1,1.414),(-1,-1,1.414)]
        while openq:
            f, g, cur, parent = heapq.heappop(openq)
            if cur in seen: continue
            seen.add(cur); came[cur] = parent
            if cur == goal:
                path = []
                n = cur
                while n is not None:
                    path.append(n); n = came[n]
                path.reverse()
                world_path = [self._world(*c) for c in path]
                return self._simplify([p1] + world_path[1:-1] + [p2])
            for dx,dy,cost in neigh:
                nb = (cur[0]+dx, cur[1]+dy)
                if self._blocked(*nb): continue
                ng = g+cost
                if ng < gscore.get(nb, 1e18):
                    gscore[nb] = ng
                    heapq.heappush(openq, (ng+h(nb), ng, nb, cur))
        raise RuntimeError(f"no path found {p1} -> {p2}")

    def _nudge(self, c, r=6):
        for rad in range(1, r+1):
            for dx in range(-rad, rad+1):
                for dy in range(-rad, rad+1):
                    nb = (c[0]+dx, c[1]+dy)
                    if not self._blocked(*nb):
                        return nb
        return c

    def _simplify(self, pts):
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts)-1:
            j = len(pts)-1
            while j > i+1 and not clear_segment_margin(
                    pts[i], pts[j], self.shelves, self.margin):
                j -= 1
            out.append(pts[j])
            i = j
        return out
