package main

import (
	"fmt"
)

func main() {

	a := 0
	m := make(map[int]int)

	for i := 0; i <= 9; i++ {

		fmt.Scan(&a)

		if v, ok := m[a]; ok {

			fmt.Print(v, " ")

		} else {
			m[a] = work(a)
			fmt.Print(m[a], " ")
		}

	}

}

func work(x int) int {

}
