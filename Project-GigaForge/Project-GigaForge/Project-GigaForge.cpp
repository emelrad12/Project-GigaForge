#include "Globals.h"
#include "Src/ComponentArray.h"

#include "Src/EntityManager.h"
#include <boost/hana.hpp>
namespace hana = boost::hana;
#include <cassert>
using namespace hana::literals;

template <typename T>
void Print(T data)
{
	std::cout << data << std::endl;
}

void p() {
	struct Fish { std::string name; };
	struct Cat { std::string name; };
	struct Dog { std::string name; };
	auto animals = hana::make_tuple(Fish{ "Nemo" }, Cat{ "Garfield" }, Dog{ "Snoopy" });
	// Access tuple elements with operator[] instead of std::get.
	Cat garfield = animals[1_c];
	// Perform high level algorithms on tuples (this is like std::transform)
	auto names = hana::transform(animals, [](auto a) {
		return a.name;
		});
	assert(hana::reverse(names) == hana::make_tuple("Snoopy", "Garfield", "Nemo"));
	auto animal_types = hana::make_tuple(hana::type_c<Fish*>, hana::type_c<Cat&>, hana::type_c<Dog>);
	auto no_pointers = hana::remove_if(animal_types, [](auto a) {
		return hana::traits::is_pointer(a);
		});
	static_assert(no_pointers == hana::make_tuple(hana::type_c<Cat&>, hana::type_c<Dog>), "");
	auto has_name = hana::is_valid([](auto&& x) -> decltype((void)x.name) {});
	static_assert(has_name(garfield), "");
	static_assert(!has_name(1), "");
	struct Person {
		BOOST_HANA_DEFINE_STRUCT(Person,
			(std::string, name),
			(int, age)
		);
	};
	// 2. Write a generic serializer (bear with std::ostream for the example)
	auto serialize = [](std::ostream& os, auto const& object) {
		hana::for_each(hana::members(object), [&](auto member) {
			os << member << std::endl;
			});
	};
	// 3. Use it
	Person john{ "John", 30 };
	serialize(std::cout, john);

}

int main()
{
	EntityManager manager = EntityManager();
	manager.AddType<int>();
	auto componentArray = manager.GetComponentArray<int>();
	#pragma omp parallel for
	for (int i = 0; i < 1000000; i++)
	{

	}
	componentArray[50] = 555;
	Print(componentArray[50]);
}
